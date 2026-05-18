#pragma once

#ifdef PULSAR_HAS_TORCH

#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace pulsar {

class RolloutStorage {
 public:
  RolloutStorage(
      int rollout_length,
      int num_agents,
      int obs_dim,
      int action_dim,
      torch::Device device,
      std::vector<std::string> head_names = {"extrinsic"},
      bool pin_memory = false);

  void append(
      int step,
      const torch::Tensor& obs,
      const torch::Tensor& episode_starts,
      const torch::Tensor& action_masks,
      const torch::Tensor& learner_active,
      const torch::Tensor& actions,
      const torch::Tensor& action_log_probs,
      const std::unordered_map<std::string, torch::Tensor>& values,
      const std::unordered_map<std::string, torch::Tensor>& rewards,
      const torch::Tensor& dones,
      const torch::Tensor& truncated,
      const torch::Tensor& bootstrap_truncated,
      const torch::Tensor& goal_positions,
      const torch::Tensor& terminal_outcome_labels,
      const torch::Tensor& terminal_observations);
  void append_slice(
      int step,
      int agent_offset,
      const torch::Tensor& obs,
      const torch::Tensor& episode_starts,
      const torch::Tensor& action_masks,
      const torch::Tensor& learner_active,
      const torch::Tensor& actions,
      const torch::Tensor& action_log_probs,
      const std::unordered_map<std::string, torch::Tensor>& values,
      const std::unordered_map<std::string, torch::Tensor>& rewards,
      const torch::Tensor& dones,
      const torch::Tensor& truncated,
      const torch::Tensor& bootstrap_truncated,
      const torch::Tensor& goal_positions,
      const torch::Tensor& terminal_outcome_labels,
      const torch::Tensor& terminal_observations);

  void set_final_values(const std::unordered_map<std::string, torch::Tensor>& final_values);
  void set_rewards_at(int step, const std::unordered_map<std::string, torch::Tensor>& rewards_in);
  [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& final_values() const;
  void clear();

  [[nodiscard]] int rollout_length() const;
  [[nodiscard]] int capacity() const;
  [[nodiscard]] int num_agents() const;

  [[nodiscard]] torch::Tensor value(const std::string& head_name) const;
  [[nodiscard]] torch::Tensor reward(const std::string& stream_name) const;
  [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& all_values() const;
  [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& all_rewards() const;

  torch::Tensor obs;
  torch::Tensor episode_starts;
  torch::Tensor action_masks;
  torch::Tensor learner_active;
  torch::Tensor actions;
  torch::Tensor action_log_probs;
  torch::Tensor dones;
  torch::Tensor truncated;
  torch::Tensor bootstrap_truncated;
  torch::Tensor goal_positions;
  torch::Tensor terminal_outcome_labels;
  torch::Tensor terminal_observations;
  torch::Tensor mode_ids;  // int8, [rollout_length, num_agents], 0=unknown 1=1v1 2=2v2 3=3v3
  std::unordered_map<std::string, torch::Tensor> final_values_;

  void set_mode_ids_slice(int step, int agent_offset, int agent_count, std::int8_t mode_id);
  void set_mode_ids_slice(int step, int agent_offset, std::int8_t mode_id);

 private:
  int rollout_length_ = 0;
  int filled_length_ = 0;
  int num_agents_ = 0;
  torch::Device device_{torch::kCPU};
  std::unordered_map<std::string, torch::Tensor> values_;
  std::unordered_map<std::string, torch::Tensor> rewards_;
};

}  // namespace pulsar

#endif
