#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>
#include "pulsar/config/config.hpp"
#include "pulsar/model/actor.hpp"
#include "pulsar/training/sac_critic.hpp"

namespace pulsar {

struct SubgoalResult {
  torch::Tensor subgoal;        // [batch, goal_dim]
  torch::Tensor planner_score;  // [batch] — for logging
  bool replanned = false;
};

class SubgoalPlanner {
 public:
  SubgoalPlanner(
      const SubgoalPlannerConfig& config,
      const GoalCriticConfig& gc_config,
      const torch::Device& device);

  // Select subgoals for the current batch.
  // goal_critic_a/b: ensemble for pessimistic reachability scoring (min-over-two).
  // sac_critic: used when sac_ready=true for leaf bootstrap (currently uses GC proxy).
  // buffer_goals: [N, goal_dim] — candidate goals from goal candidate buffer.
  // task_goals: [batch, goal_dim] — opponent net.
  SubgoalResult select_subgoals(
      GoalCritic& goal_critic_a,
      GoalCritic& goal_critic_b,
      SACCritic& sac_critic,
      const torch::Tensor& current_features,  // [batch, feature_dim]
      const torch::Tensor& current_actions,   // [batch] int64
      const torch::Tensor& buffer_goals,      // [N, goal_dim]
      const torch::Tensor& task_goals,        // [batch, goal_dim]
      bool sac_ready = false);

  // Check for early abort (subgoal reached or progress stalled).
  bool should_replan(
      GoalCritic& goal_critic_a,
      GoalCritic& goal_critic_b,
      const torch::Tensor& current_features,
      const torch::Tensor& current_actions,
      const torch::Tensor& current_subgoal,
      const torch::Tensor& recent_scores);   // [history_window, batch]

  [[nodiscard]] const torch::Tensor& current_subgoal() const { return current_subgoal_; }
  void increment_step() { ++steps_since_replan_; }

 private:
  // Score candidates using GoalCritic (reachability) + GoalCritic goal-goal proxy (progress)
  // + uncertainty penalty.
  // Returns [batch, N] scores.
  torch::Tensor score_candidates(
      GoalCritic& goal_critic_a,
      GoalCritic& goal_critic_b,
      const torch::Tensor& features,   // [batch, feature_dim]
      const torch::Tensor& actions,    // [batch]
      const torch::Tensor& candidates, // [N, goal_dim]
      const torch::Tensor& task_goals);

  SubgoalPlannerConfig config_;
  GoalCriticConfig gc_config_;
  torch::Device device_;
  int steps_since_replan_ = 0;
  torch::Tensor current_subgoal_;
};

}  // namespace pulsar

#endif
