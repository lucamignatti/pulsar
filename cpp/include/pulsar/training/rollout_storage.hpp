#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

#include "pulsar/model/latent_future_actor.hpp"

namespace pulsar {

class RolloutStorage {
 public:
  RolloutStorage(
      int rollout_length,
      int num_agents,
      int obs_dim,
      int action_dim,
      int candidate_count,
      torch::Device device);

  void append(
      int step,
      const torch::Tensor& raw_obs,
      const torch::Tensor& obs,
      const torch::Tensor& episode_starts,
      const torch::Tensor& action_masks,
      const torch::Tensor& learner_active,
      const torch::Tensor& executed_actions,
      const torch::Tensor& candidate_actions,
      const torch::Tensor& candidate_log_probs,
      const torch::Tensor& trajectory_ids,
      const torch::Tensor& dones,
      const torch::Tensor& terminal_outcomes);

  void set_final_observation(const torch::Tensor& raw_obs);
  void set_initial_state(const ContinuumState& state);
  [[nodiscard]] ContinuumState initial_state_for_agents(const torch::Tensor& agent_indices) const;

  [[nodiscard]] int rollout_length() const;
  [[nodiscard]] int num_agents() const;
  [[nodiscard]] int candidate_count() const;

  ContinuumState initial_state{};
  torch::Tensor raw_obs;
  torch::Tensor final_raw_obs;
  torch::Tensor obs;
  torch::Tensor episode_starts;
  torch::Tensor action_masks;
  torch::Tensor learner_active;
  torch::Tensor executed_actions;
  torch::Tensor candidate_actions;
  torch::Tensor candidate_log_probs;
  torch::Tensor trajectory_ids;
  torch::Tensor dones;
  torch::Tensor terminal_outcomes;

 private:
  int rollout_length_ = 0;
  int num_agents_ = 0;
  int candidate_count_ = 0;
  torch::Device device_{torch::kCPU};
};

}  // namespace pulsar

#endif
