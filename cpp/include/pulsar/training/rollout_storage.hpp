#pragma once

#ifdef PULSAR_HAS_TORCH

#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "pulsar/model/ppo_actor.hpp"

namespace pulsar {

class RolloutStorage {
 public:
  RolloutStorage(
      int rollout_length,
      int num_agents,
      int obs_dim,
      int action_dim,
      int encoder_dim,
      torch::Device device,
      std::vector<std::string> head_names = {"extrinsic", "curiosity", "learning_progress", "controllability"});

  void append(
      int step,
      const torch::Tensor& raw_obs,
      const torch::Tensor& obs,
      const torch::Tensor& encoded,
      const torch::Tensor& episode_starts,
      const torch::Tensor& action_masks,
      const torch::Tensor& learner_active,
      const torch::Tensor& actions,
      const torch::Tensor& action_log_probs,
      const std::unordered_map<std::string, torch::Tensor>& values,
      const std::unordered_map<std::string, torch::Tensor>& rewards,
      const torch::Tensor& dones);

  void set_final_observation(const torch::Tensor& raw_obs);
  void set_final_encoded(const torch::Tensor& encoded);
  void set_final_values(const std::unordered_map<std::string, torch::Tensor>& final_values);
  void set_rewards_at(int step, const std::unordered_map<std::string, torch::Tensor>& rewards_in);
  [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& final_values() const;
  void set_initial_state(const ContinuumState& state);
  [[nodiscard]] ContinuumState initial_state_for_agents(const torch::Tensor& agent_indices) const;

  [[nodiscard]] int rollout_length() const;
  [[nodiscard]] int num_agents() const;

  [[nodiscard]] torch::Tensor value(const std::string& head_name) const;
  [[nodiscard]] torch::Tensor reward(const std::string& stream_name) const;
  [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& all_values() const;
  [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& all_rewards() const;

  ContinuumState initial_state{};
  torch::Tensor raw_obs;
  torch::Tensor final_raw_obs;
  torch::Tensor obs;
  torch::Tensor encoded;
  torch::Tensor final_encoded;
  torch::Tensor episode_starts;
  torch::Tensor action_masks;
  torch::Tensor learner_active;
  torch::Tensor actions;
  torch::Tensor action_log_probs;
  torch::Tensor dones;
  std::unordered_map<std::string, torch::Tensor> final_values_;

 private:
  int rollout_length_ = 0;
  int num_agents_ = 0;
  torch::Device device_{torch::kCPU};
  std::unordered_map<std::string, torch::Tensor> values_;
  std::unordered_map<std::string, torch::Tensor> rewards_;
};

}  // namespace pulsar

#endif