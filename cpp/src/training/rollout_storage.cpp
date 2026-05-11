#include "pulsar/training/rollout_storage.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <stdexcept>

#include "pulsar/training/ppo_math.hpp"

namespace pulsar {

RolloutStorage::RolloutStorage(
    int rollout_length,
    int num_agents,
    int obs_dim,
    int action_dim,
    torch::Device device,
    std::vector<std::string> head_names)
    : rollout_length_(rollout_length),
      num_agents_(num_agents),
      device_(device) {
  obs = torch::zeros({rollout_length, num_agents, obs_dim}, device);
  episode_starts = torch::zeros({rollout_length, num_agents}, device);
  action_masks = torch::zeros(
      {rollout_length, num_agents, action_dim},
      torch::TensorOptions().dtype(torch::kUInt8).device(device));
  learner_active = torch::zeros({rollout_length, num_agents}, device);
  actions = torch::zeros({rollout_length, num_agents}, torch::TensorOptions().dtype(torch::kLong).device(device));
  action_log_probs = torch::zeros({rollout_length, num_agents}, device);
  dones = torch::zeros({rollout_length, num_agents}, device);
  truncated = torch::zeros({rollout_length, num_agents}, device);
  bootstrap_truncated = torch::zeros({rollout_length, num_agents}, device);
  goal_positions = torch::zeros({rollout_length, num_agents, 3}, device);
  terminal_outcome_labels = torch::full(
      {rollout_length, num_agents}, 2,
      torch::TensorOptions().dtype(torch::kInt64).device(device));

  for (const auto& name : head_names) {
    values_[name] = torch::zeros({rollout_length, num_agents}, device);
    rewards_[name] = torch::zeros({rollout_length, num_agents}, device);
  }
}

void RolloutStorage::append(
    int step,
    const torch::Tensor& obs_in,
    const torch::Tensor& episode_starts_in,
    const torch::Tensor& action_masks_in,
    const torch::Tensor& learner_active_in,
    const torch::Tensor& actions_in,
    const torch::Tensor& action_log_probs_in,
    const std::unordered_map<std::string, torch::Tensor>& values_in,
    const std::unordered_map<std::string, torch::Tensor>& rewards_in,
    const torch::Tensor& dones_in,
    const torch::Tensor& truncated_in,
    const torch::Tensor& bootstrap_truncated_in,
    const torch::Tensor& goal_positions_in,
    const torch::Tensor& terminal_outcome_labels_in) {
  if (step < 0 || step >= rollout_length_) {
    throw std::out_of_range("RolloutStorage::append step is outside rollout capacity.");
  }
  obs[step].copy_(obs_in.detach());
  episode_starts[step].copy_(episode_starts_in.detach());
  action_masks[step].copy_(action_masks_in.detach());
  learner_active[step].copy_(learner_active_in.detach());
  actions[step].copy_(actions_in.detach());
  action_log_probs[step].copy_(action_log_probs_in.detach());
  dones[step].copy_(dones_in.detach());
  truncated[step].copy_(truncated_in.detach());
  bootstrap_truncated[step].copy_(bootstrap_truncated_in.detach());
  goal_positions[step].copy_(goal_positions_in.detach());
  terminal_outcome_labels[step].copy_(terminal_outcome_labels_in.detach());

  for (const auto& [name, tensor] : values_in) {
    auto it = values_.find(name);
    if (it != values_.end()) {
      it->second[step].copy_(tensor.detach());
    }
  }
  for (const auto& [name, tensor] : rewards_in) {
    auto it = rewards_.find(name);
    if (it != rewards_.end()) {
      it->second[step].copy_(tensor.detach());
    }
  }
  filled_length_ = std::max(filled_length_, step + 1);
}

void RolloutStorage::append_slice(
    int step,
    int agent_offset,
    const torch::Tensor& obs_in,
    const torch::Tensor& episode_starts_in,
    const torch::Tensor& action_masks_in,
    const torch::Tensor& learner_active_in,
    const torch::Tensor& actions_in,
    const torch::Tensor& action_log_probs_in,
    const std::unordered_map<std::string, torch::Tensor>& values_in,
    const std::unordered_map<std::string, torch::Tensor>& rewards_in,
    const torch::Tensor& dones_in,
    const torch::Tensor& truncated_in,
    const torch::Tensor& bootstrap_truncated_in,
    const torch::Tensor& goal_positions_in,
    const torch::Tensor& terminal_outcome_labels_in) {
  if (step < 0 || step >= rollout_length_) {
    throw std::out_of_range("RolloutStorage::append_slice step is outside rollout capacity.");
  }
  const int agent_count = static_cast<int>(obs_in.size(0));
  if (agent_offset < 0 || agent_count < 0 || agent_offset + agent_count > num_agents_) {
    throw std::out_of_range("RolloutStorage::append_slice agent range is outside rollout capacity.");
  }

  obs[step].narrow(0, agent_offset, agent_count).copy_(obs_in.detach());
  episode_starts[step].narrow(0, agent_offset, agent_count).copy_(episode_starts_in.detach());
  action_masks[step].narrow(0, agent_offset, agent_count).copy_(action_masks_in.detach());
  learner_active[step].narrow(0, agent_offset, agent_count).copy_(learner_active_in.detach());
  actions[step].narrow(0, agent_offset, agent_count).copy_(actions_in.detach());
  action_log_probs[step].narrow(0, agent_offset, agent_count).copy_(action_log_probs_in.detach());
  dones[step].narrow(0, agent_offset, agent_count).copy_(dones_in.detach());
  truncated[step].narrow(0, agent_offset, agent_count).copy_(truncated_in.detach());
  bootstrap_truncated[step].narrow(0, agent_offset, agent_count).copy_(bootstrap_truncated_in.detach());
  goal_positions[step].narrow(0, agent_offset, agent_count).copy_(goal_positions_in.detach());
  terminal_outcome_labels[step].narrow(0, agent_offset, agent_count).copy_(terminal_outcome_labels_in.detach());

  for (const auto& [name, tensor] : values_in) {
    auto it = values_.find(name);
    if (it != values_.end()) {
      it->second[step].narrow(0, agent_offset, agent_count).copy_(tensor.detach());
    }
  }
  for (const auto& [name, tensor] : rewards_in) {
    auto it = rewards_.find(name);
    if (it != rewards_.end()) {
      it->second[step].narrow(0, agent_offset, agent_count).copy_(tensor.detach());
    }
  }
  filled_length_ = std::max(filled_length_, step + 1);
}

void RolloutStorage::set_final_values(
    const std::unordered_map<std::string, torch::Tensor>& final_values) {
  for (const auto& [name, tensor] : final_values) {
    final_values_[name] = tensor.detach().clone();
  }
}

const std::unordered_map<std::string, torch::Tensor>& RolloutStorage::final_values() const {
  return final_values_;
}

void RolloutStorage::clear() {
  filled_length_ = 0;
  final_values_.clear();
}

int RolloutStorage::rollout_length() const {
  return filled_length_;
}

int RolloutStorage::capacity() const {
  return rollout_length_;
}

int RolloutStorage::num_agents() const {
  return num_agents_;
}

torch::Tensor RolloutStorage::value(const std::string& head_name) const {
  auto it = values_.find(head_name);
  if (it != values_.end()) {
    return it->second;
  }
  return values_.at("extrinsic");
}

torch::Tensor RolloutStorage::reward(const std::string& stream_name) const {
  auto it = rewards_.find(stream_name);
  if (it != rewards_.end()) {
    return it->second;
  }
  return rewards_.at("extrinsic");
}

const std::unordered_map<std::string, torch::Tensor>& RolloutStorage::all_values() const {
  return values_;
}

const std::unordered_map<std::string, torch::Tensor>& RolloutStorage::all_rewards() const {
  return rewards_;
}

void RolloutStorage::set_rewards_at(
    int step,
    const std::unordered_map<std::string, torch::Tensor>& rewards_in) {
  for (const auto& [name, tensor] : rewards_in) {
    auto it = rewards_.find(name);
    if (it != rewards_.end()) {
      it->second[step].copy_(tensor.detach());
    }
  }
}

}  // namespace pulsar

#endif
