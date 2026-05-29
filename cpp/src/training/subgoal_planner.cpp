#include "pulsar/training/subgoal_planner.hpp"

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

namespace pulsar {

SubgoalPlanner::SubgoalPlanner(
    const SubgoalPlannerConfig& config,
    const GoalCriticConfig& gc_config,
    const torch::Device& device)
    : config_(config), gc_config_(gc_config), device_(device) {}

torch::Tensor SubgoalPlanner::score_candidates(
    GoalCritic& goal_critic_a,
    GoalCritic& goal_critic_b,
    const torch::Tensor& features,
    const torch::Tensor& actions,
    const torch::Tensor& candidates,
    const torch::Tensor& task_goals) {

  const int64_t batch = features.size(0);
  const int64_t N = candidates.size(0);

  if (N == 0) {
    return torch::full({batch, 1}, -1e9F, features.options());
  }

  // Expand features and actions to [batch*N, ...]
  // features: [batch, feat_dim] → [batch, N, feat_dim] → [batch*N, feat_dim]
  const torch::Tensor feat_exp = features.unsqueeze(1)
      .expand({batch, N, features.size(1)})
      .reshape({batch * N, features.size(1)});
  const torch::Tensor act_exp = actions.unsqueeze(1)
      .expand({batch, N})
      .reshape({batch * N});
  // candidates: [N, goal_dim] → [batch, N, goal_dim] → [batch*N, goal_dim]
  const torch::Tensor cand_exp = candidates.unsqueeze(0)
      .expand({batch, N, candidates.size(1)})
      .reshape({batch * N, candidates.size(1)});
  // task goals: [batch, goal_dim] → [batch*N, goal_dim]
  const torch::Tensor task_exp = task_goals.unsqueeze(1)
      .expand({batch, N, task_goals.size(1)})
      .reshape({batch * N, task_goals.size(1)});

  torch::NoGradGuard no_grad;

  // Reachability: GoalCritic returns -L2^2, higher = more reachable
  const torch::Tensor reach_a = goal_critic_a->forward(feat_exp, act_exp, cand_exp).reshape({batch, N});
  const torch::Tensor reach_b = goal_critic_b->forward(feat_exp, act_exp, cand_exp).reshape({batch, N});
  const torch::Tensor reach   = torch::minimum(reach_a, reach_b);  // pessimistic [batch, N]

  // Progress: GoalCritic goal-to-goal distance proxy
  // sg_emb: [batch*N, emb_dim], task_emb: [batch*N, emb_dim]
  const torch::Tensor sg_emb   = goal_critic_a->goal_embedding(cand_exp);
  const torch::Tensor task_emb = goal_critic_a->goal_embedding(task_exp);
  const torch::Tensor progress = -(sg_emb - task_emb).square().sum(-1).reshape({batch, N});

  // Uncertainty: ensemble disagreement
  const torch::Tensor uncertainty = (reach_a - reach_b).abs();  // [batch, N]

  const float alpha  = config_.reachability_weight;
  const float lambda = config_.uncertainty_penalty;
  torch::Tensor score = alpha * reach + (1.0F - alpha) * progress - lambda * uncertainty;

  // Hard floor: reject candidates with insufficient reachability
  const float min_reach = config_.min_reachability;
  score = torch::where(reach > -min_reach, score,
      torch::full_like(score, -1e9F));

  return score;  // [batch, N]
}

SubgoalResult SubgoalPlanner::select_subgoals(
    GoalCritic& goal_critic_a,
    GoalCritic& goal_critic_b,
    SACCritic& sac_critic,
    const torch::Tensor& current_features,
    const torch::Tensor& current_actions,
    const torch::Tensor& buffer_goals,
    const torch::Tensor& task_goals,
    bool /*sac_ready*/) {

  (void)sac_critic;  // Progress term always uses GoalCritic proxy per spec

  if (!buffer_goals.defined() || buffer_goals.size(0) == 0) {
    SubgoalResult result;
    result.replanned = false;
    return result;
  }

  // Subsample candidates to candidate_buffer_size
  const int64_t total_candidates = buffer_goals.size(0);
  torch::Tensor candidates = buffer_goals;
  if (total_candidates > config_.candidate_buffer_size) {
    const torch::Tensor perm = torch::randperm(total_candidates,
        torch::TensorOptions().dtype(torch::kLong).device(buffer_goals.device()))
        .narrow(0, 0, config_.candidate_buffer_size);
    candidates = buffer_goals.index_select(0, perm);
  }

  const torch::Tensor scores = score_candidates(
      goal_critic_a, goal_critic_b,
      current_features, current_actions, candidates, task_goals);  // [batch, N]

  // Argmax per batch element
  const torch::Tensor best_idx = scores.argmax(1);  // [batch]
  const torch::Tensor selected = candidates.index_select(0, best_idx);  // [batch, goal_dim]
  const torch::Tensor best_scores = scores.gather(1, best_idx.unsqueeze(1)).squeeze(1);

  current_subgoal_ = selected;
  steps_since_replan_ = 0;

  SubgoalResult result;
  result.subgoal = selected;
  result.planner_score = best_scores;
  result.replanned = true;
  return result;
}

bool SubgoalPlanner::should_replan(
    GoalCritic& goal_critic_a,
    GoalCritic& goal_critic_b,
    const torch::Tensor& current_features,
    const torch::Tensor& current_actions,
    const torch::Tensor& current_subgoal,
    const torch::Tensor& recent_scores) {

  if (steps_since_replan_ < 2 || !current_subgoal.defined()) return false;

  const int64_t hist_len = recent_scores.size(0);
  if (hist_len == 0) return false;

  // Reachability score toward current subgoal
  torch::NoGradGuard no_grad;
  const torch::Tensor score_a = goal_critic_a->forward(
      current_features, current_actions, current_subgoal).detach();
  const torch::Tensor score_b = goal_critic_b->forward(
      current_features, current_actions, current_subgoal).detach();
  const torch::Tensor current_score = torch::minimum(score_a, score_b);

  // Check if subgoal reached: score close to 0 (L2^2 ≈ 0 → score ≈ 0)
  if ((current_score > -0.01F).any().item<bool>()) return true;

  // Check stalled progress over last half commit horizon
  const int window = std::max(2, config_.commit_horizon / 2);
  if (hist_len >= static_cast<int64_t>(window)) {
    const torch::Tensor old_score = recent_scores[hist_len - window].mean();
    const torch::Tensor new_score = recent_scores[hist_len - 1].mean();
    if ((new_score <= old_score).item<bool>()) return true;
  }

  return false;
}

}  // namespace pulsar

#endif
