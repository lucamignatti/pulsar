#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

#include "pulsar/model/actor_critic.hpp"

namespace pulsar {

class RolloutStorage {
 public:
  RolloutStorage(
      int rollout_length,
      int num_agents,
      int obs_dim,
      int action_dim,
      torch::Device device);

  void append(
      int step,
      const torch::Tensor& obs,
      const torch::Tensor& episode_starts,
      const torch::Tensor& action_masks,
      const torch::Tensor& learner_active,
      const torch::Tensor& actions,
      const torch::Tensor& log_probs,
      const torch::Tensor& rewards,
      const torch::Tensor& dones,
      const torch::Tensor& sampled_values);

  void compute_returns_and_advantages(
      const torch::Tensor& last_sampled_value,
      float gamma,
      float gae_lambda);
  void set_initial_state(const ContinuumState& state);
  [[nodiscard]] ContinuumState initial_state_for_agents(const torch::Tensor& agent_indices) const;

  [[nodiscard]] int rollout_length() const;
  [[nodiscard]] int num_agents() const;

  ContinuumState initial_state{};
  torch::Tensor obs;
  torch::Tensor episode_starts;
  torch::Tensor action_masks;
  torch::Tensor learner_active;
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
