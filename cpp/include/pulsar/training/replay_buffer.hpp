#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>
#include "pulsar/config/config.hpp"

namespace pulsar {

// Off-policy replay buffer for SAC critic training with HER relabeling.
// Stores achieved_goal (ball pos after action) and conditioned_goal (agent's current goal)
// separately. Reward is computed entirely at sample time: r=1 if achieved_goal ≈ conditioned_goal.
// r_terminal is NEVER stored here — this ensures SAC trains on a clean goal-reaching MDP.
class ReplayBuffer {
 public:
  ReplayBuffer(
      const ReplayBufferConfig& config,
      int obs_dim,
      int action_dim,
      int goal_dim,
      const torch::Device& host_device = torch::Device(torch::kCPU));

  // Write one PPO step's worth of transitions.
  // achieved_goal: ball position AFTER the action (post-step ball pos = host_goal_positions())
  // conditioned_goal: what the agent was trying to reach (subgoal or task_goal_host)
  // r_terminal is intentionally absent — reward computed at sample time from goals.
  void push(
      const torch::Tensor& obs,              // [batch, obs_dim] — pre-step obs
      const torch::Tensor& actions,          // [batch] int64
      const torch::Tensor& achieved_goal,    // [batch, goal_dim] — ball pos after step
      const torch::Tensor& conditioned_goal, // [batch, goal_dim] — subgoal or task goal
      const torch::Tensor& next_obs,         // [batch, obs_dim] — post-step obs
      const torch::Tensor& dones,            // [batch] float
      const torch::Tensor& episode_ids);     // [batch] int64 — incremented on reset

  struct TransitionBatch {
    torch::Tensor obs;
    torch::Tensor actions;
    torch::Tensor rewards;      // {0, 1} — computed from HER relabeling
    torch::Tensor next_obs;
    torch::Tensor dones;
    torch::Tensor goals;        // conditioned_goal after HER relabeling
  };
  TransitionBatch sample_transitions(int batch_size);

  // Segment batch for LQL (segments respect episode boundaries).
  struct SegmentBatch {
    torch::Tensor obs;           // [L, B, obs_dim]
    torch::Tensor actions;       // [L, B] int64
    torch::Tensor rewards;       // [L, B] — HER-relabeled, {0, 1}
    torch::Tensor next_obs;      // [L, B, obs_dim]
    torch::Tensor dones;         // [L, B]
    torch::Tensor goals;         // [L, B, goal_dim]
    torch::Tensor episode_ends;  // [L, B] — 1 at episode boundaries (for LQL boundary masking)
  };
  SegmentBatch sample_segments(int num_segments);

  [[nodiscard]] int size() const;
  [[nodiscard]] bool ready() const;

 private:
  // HER relabeling: for a batch of sampled transitions, replace conditioned_goal with a
  // future achieved_goal from the same episode for her_relabel_fraction of transitions.
  // Returns relabeled (goals, rewards). rewards = 1 if achieved_goal ≈ relabeled goal.
  std::pair<torch::Tensor, torch::Tensor> her_relabel_batch(
      const torch::Tensor& achieved_goals,    // [N, goal_dim]
      const torch::Tensor& conditioned_goals, // [N, goal_dim]
      const torch::Tensor& episode_ids,       // [N]
      const torch::Tensor& timestep_ids,      // [N]
      const torch::Tensor& sampled_indices);  // [N] — buffer positions

  torch::Tensor compute_goal_reward(
      const torch::Tensor& achieved_goals,
      const torch::Tensor& target_goals) const;

  ReplayBufferConfig config_;
  int obs_dim_;
  int action_dim_;
  int goal_dim_;

  // Circular buffers — stored CPU, pinned for fast GPU transfer
  torch::Tensor buf_obs_;
  torch::Tensor buf_actions_;
  torch::Tensor buf_next_obs_;
  torch::Tensor buf_dones_;
  torch::Tensor buf_achieved_goals_;
  torch::Tensor buf_conditioned_goals_;
  torch::Tensor buf_episode_ids_;
  torch::Tensor buf_timestep_ids_;

  int head_ = 0;    // next write position
  int filled_ = 0;  // number of valid entries
};

}  // namespace pulsar

#endif
