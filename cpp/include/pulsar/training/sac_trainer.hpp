#pragma once

#ifdef PULSAR_HAS_TORCH

#include <memory>
#include <torch/torch.h>
#include "pulsar/config/config.hpp"
#include "pulsar/training/replay_buffer.hpp"
#include "pulsar/training/sac_critic.hpp"

namespace pulsar {

struct SACTrainerLosses {
  double td_loss = 0.0;
  double lql_ub_loss = 0.0;
  double lql_lb_loss = 0.0;
  double total_loss = 0.0;
  double mean_q_value = 0.0;
  double max_q_value = 0.0;
};

class SACTrainer {
 public:
  SACTrainer(const SACCriticConfig& config, const torch::Device& device);

  SACTrainerLosses update(
      SACCritic& critic,
      const ReplayBuffer::SegmentBatch& batch);

  // Must be called once before the first update to initialize the optimizer.
  void init_optimizer(SACCritic& critic);

 private:
  torch::Tensor compute_td_loss(
      SACCritic& critic,
      const torch::Tensor& obs,      // [N, obs_dim]
      const torch::Tensor& actions,  // [N]
      const torch::Tensor& rewards,  // [N]
      const torch::Tensor& next_obs, // [N, obs_dim]
      const torch::Tensor& dones,    // [N]
      const torch::Tensor& goals);   // [N, goal_dim]

  // LQL losses computed over segments [L, B, ...]
  torch::Tensor compute_lql_losses(
      SACCritic& critic,
      const ReplayBuffer::SegmentBatch& batch);

  SACCriticConfig config_;
  torch::Device device_;
  std::unique_ptr<torch::optim::Adam> optimizer_;
  bool optimizer_initialized_ = false;
};

}  // namespace pulsar

#endif
