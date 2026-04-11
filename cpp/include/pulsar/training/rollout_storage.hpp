#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

namespace pulsar {

class RolloutStorage {
 public:
  RolloutStorage(
      int rollout_length,
      int num_agents,
      int obs_dim,
      torch::Device device);

  void append(
      int step,
      const torch::Tensor& obs,
      const torch::Tensor& episode_starts,
      const torch::Tensor& actions,
      const torch::Tensor& log_probs,
      const torch::Tensor& rewards,
      const torch::Tensor& dones,
      const torch::Tensor& sampled_values);

  void compute_returns_and_advantages(
      const torch::Tensor& last_sampled_value,
      float gamma,
      float gae_lambda);

  [[nodiscard]] int rollout_length() const;
  [[nodiscard]] int num_agents() const;

  torch::Tensor obs;
  torch::Tensor episode_starts;
  torch::Tensor actions;
  torch::Tensor log_probs;
  torch::Tensor rewards;
  torch::Tensor dones;
  torch::Tensor sampled_values;
  torch::Tensor returns;
  torch::Tensor advantages;

 private:
  int rollout_length_ = 0;
  int num_agents_ = 0;
};

}  // namespace pulsar

#endif
