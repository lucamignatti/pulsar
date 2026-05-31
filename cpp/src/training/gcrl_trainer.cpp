#include "pulsar/training/gcrl_trainer.hpp"

#include <algorithm>
#include <limits>
#include <cmath>

#include "pulsar/training/ppo_math.hpp"
#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

torch::Tensor sample_masked_gumbel_softmax(
    const torch::Tensor& logits,
    const torch::Tensor& masks,
    float temperature) {
  const torch::Tensor masked_logits = apply_action_mask_to_logits(logits, masks);
  const torch::Tensor uniform = torch::rand_like(masked_logits).clamp(1.0e-6F, 1.0F - 1.0e-6F);
  const torch::Tensor gumbel = -torch::log(-torch::log(uniform));
  const torch::Tensor y_soft = torch::softmax((masked_logits + gumbel) / std::max(temperature, 1.0e-3F), -1);
  // Straight-Through (paper spec): hard one-hot in the forward pass so the critic
  // sees the same action representation it is trained on (real actions are hard
  // one-hots), while gradients flow through the soft distribution in the backward
  // pass. Feeding soft probs forward would be a train/eval mismatch for the critic.
  const torch::Tensor index = y_soft.argmax(-1, /*keepdim=*/true);
  // zeros_like is called on the detached soft tensor so scatter_ operates on a
  // plain non-grad tensor; y_hard has no autograd history.
  const torch::Tensor y_hard = torch::zeros_like(y_soft.detach()).scatter_(-1, index, 1.0F);
  // ST: stop gradient through the (y_hard - y_soft) correction so that only
  // y_soft carries the backward pass (matching the soft distribution gradient).
  return y_soft + (y_hard - y_soft.detach());
}

torch::Tensor goal_actor_critic_loss(
    GoalCritic& goal_critic,
    const torch::Tensor& features,
    const torch::Tensor& logits,
    const torch::Tensor& masks,
    const torch::Tensor& future_goals,
    int contrastive_batch_size) {
  const auto active_count = features.size(0);
  if (active_count <= 0) {
    return torch::zeros({}, logits.options().dtype(torch::kFloat32));
  }

  torch::Tensor selected_features = features;
  torch::Tensor selected_logits = logits;
  torch::Tensor selected_masks = masks;
  torch::Tensor selected_goals = future_goals;
  const int bounded_batch = std::max(1, contrastive_batch_size);
  if (active_count > static_cast<c10::IntArrayRef::value_type>(bounded_batch)) {
    const torch::Tensor idx = torch::randperm(
        active_count,
        torch::TensorOptions().dtype(torch::kLong).device(logits.device()))
        .narrow(0, 0, bounded_batch);
    selected_features = selected_features.index({idx});
    selected_logits = selected_logits.index({idx});
    selected_masks = selected_masks.index({idx});
    selected_goals = selected_goals.index({idx});
  }

  const torch::Tensor action_probs = sample_masked_gumbel_softmax(
      selected_logits,
      selected_masks,
      1.0F);
  ModuleRequiresGradGuard freeze_goal_critic(*goal_critic, false);
  return -goal_critic->forward(
      selected_features.detach(),
      action_probs,
      selected_goals.detach()).to(torch::kFloat32).mean();
}

GCRLTrainer::GCRLTrainer(
    const GoalCriticConfig& config,
    const torch::Device& device)
    : config_(config), device_(device) {}

void GCRLTrainer::compute_gcrl_losses(
    GoalCritic& goal_critic,
    const torch::Tensor& active_features,
    const torch::Tensor& active_actions,
    const torch::Tensor& active_logits,
    const torch::Tensor& active_masks,
    const torch::Tensor& active_future_goal_pos,
    const torch::Tensor& active_actor_goal_pos,
    bool compute_critic_loss,
    bool compute_actor_loss,
    torch::Tensor& goal_loss,
    torch::Tensor& actor_goal_loss,
    torch::Tensor& goal_score) {
  PULSAR_TRACE_SCOPE_CAT("gcrl_trainer", "compute_gcrl_losses");

  const auto active_count = active_features.size(0);
  const int cb_size = config_.contrastive_batch_size;

  torch::Tensor selected_features = active_features;
  torch::Tensor selected_actions = active_actions;
  torch::Tensor selected_logits = active_logits;
  torch::Tensor selected_masks = active_masks;
  torch::Tensor selected_future_goal_pos = active_future_goal_pos;

  if (active_count > cb_size) {
    const torch::Tensor idx = torch::randperm(active_count, active_actions.options()).narrow(0, 0, cb_size);
    selected_features = selected_features.index({idx});
    selected_actions = selected_actions.index({idx});
    selected_logits = selected_logits.index({idx});
    selected_masks = selected_masks.index({idx});
    selected_future_goal_pos = selected_future_goal_pos.index({idx});
  }

  if (compute_critic_loss) {
    // Features are NOT detached: the contrastive InfoNCE objective is what
    // trains the shared encoder's φ(o,a) representation (paper's core mechanism).
    torch::Tensor sa_emb = goal_critic->sa_embedding(selected_features, selected_actions);
    torch::Tensor g_emb = goal_critic->goal_embedding(selected_future_goal_pos);
    goal_loss = compute_symmetric_infonce_loss(
        compute_pairwise_negative_l2_logits(sa_emb, g_emb, config_.temperature),
        config_.logsumexp_penalty_coeff);
    goal_score = -((sa_emb.detach() - g_emb.detach()).square().sum(-1).clamp_min(1.0e-8F)).mean();
  }

  if (compute_actor_loss) {
    actor_goal_loss = goal_actor_critic_loss(
        goal_critic,
        active_features,
        active_logits,
        active_masks,
        active_actor_goal_pos,
        config_.contrastive_batch_size);
  }
}

}  // namespace pulsar

#endif
