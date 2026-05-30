#pragma once

#ifdef PULSAR_HAS_TORCH

#include <mutex>
#include <torch/torch.h>
#include "pulsar/config/config.hpp"

namespace pulsar {

// Off-policy episode buffer for CRL training with hindsight relabeling.
// Stores complete episodes. Goals are relabeled at sample time by picking
// a uniformly random future timestep within the same episode — no cross-episode
// ID tracking or cumsum tricks needed.
class ReplayBuffer {
 public:
  ReplayBuffer(
      const ReplayBufferConfig& config,
      int total_agents,
      int obs_dim,
      int action_dim,
      int goal_dim,
      int car_goal_dim,
      const torch::Device& host_device = torch::Device(torch::kCPU));

  // Called once per environment step for all agents simultaneously.
  // Accumulates per-agent data; flushes a complete episode to the circular
  // buffer whenever done[i]==1 for agent i.
  // agent_offset is the global index of batch[0] — used when called per-shard.
  void push_step(
      const torch::Tensor& obs,        // [batch_agents, obs_dim]
      const torch::Tensor& actions,    // [batch_agents] int64
      const torch::Tensor& ball_goals, // [batch_agents, goal_dim]   — ball pos after step
      const torch::Tensor& car_goals,  // [batch_agents, car_goal_dim] — car pos after step
      const torch::Tensor& dones,      // [batch_agents] float — flush on done==1
      int agent_offset = 0);

  // CRL batch: obs + actions + hindsight-relabeled future goals for both heads.
  struct CRLBatch {
    torch::Tensor obs;               // [B, obs_dim]
    torch::Tensor actions;           // [B] int64
    torch::Tensor future_ball_goals; // [B, goal_dim]
    torch::Tensor future_car_goals;  // [B, car_goal_dim]
  };
  CRLBatch sample_crl_batch(int batch_size);

  [[nodiscard]] int size() const;   // number of valid episodes
  [[nodiscard]] bool ready() const;

  // Legacy stub — kept so SACTrainer (not used in CRL mode) still compiles.
  struct SegmentBatch {
    torch::Tensor obs, actions, rewards, next_obs, dones, goals, episode_ends;
  };

 private:
  // Flush agent i's accumulation buffer as one complete episode.
  // Caller must hold ep_mutex_.
  void flush_agent_locked(int agent_idx);

  ReplayBufferConfig config_;
  int total_agents_;
  int obs_dim_;
  int action_dim_;
  int goal_dim_;
  int car_goal_dim_;
  int max_ep_len_;

  // Circular episode buffer — CPU pinned for fast GPU transfer.
  torch::Tensor ep_obs_;        // [max_episodes, max_ep_len, obs_dim]
  torch::Tensor ep_actions_;    // [max_episodes, max_ep_len]   int64
  torch::Tensor ep_ball_goals_; // [max_episodes, max_ep_len, goal_dim]
  torch::Tensor ep_car_goals_;  // [max_episodes, max_ep_len, car_goal_dim]
  torch::Tensor ep_lengths_;    // [max_episodes]               int32

  int head_   = 0;  // next write position (episode index)
  int filled_ = 0;  // number of valid episodes

  mutable std::mutex ep_mutex_;

  // Per-agent accumulation buffers for in-progress episodes (no lock needed —
  // only the collector thread writes these).
  torch::Tensor acc_obs_;        // [total_agents, max_ep_len, obs_dim]
  torch::Tensor acc_actions_;    // [total_agents, max_ep_len]
  torch::Tensor acc_ball_goals_; // [total_agents, max_ep_len, goal_dim]
  torch::Tensor acc_car_goals_;  // [total_agents, max_ep_len, car_goal_dim]
  torch::Tensor acc_step_;       // [total_agents]  int32 — steps written so far
};

}  // namespace pulsar

#endif
