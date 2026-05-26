#include "pulsar/training/rollout_storage.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <stdexcept>

#include "pulsar/training/ppo_math.hpp"

namespace pulsar {
namespace {

void copy_detached(torch::Tensor destination, const torch::Tensor& source) {
  destination.copy_(source.detach(), /*non_blocking=*/true);
}

}  // namespace

RolloutStorage::RolloutStorage(
    int rollout_length,
    int num_agents,
    int obs_dim,
    int action_dim,
    int encoder_dim,
    torch::Device device,
    std::vector<std::string> head_names,
    bool pin_memory)
    : rollout_length_(rollout_length),
      num_agents_(num_agents),
      device_(device) {
  auto opts = torch::TensorOptions().device(device);
  if (pin_memory) {
    opts = opts.pinned_memory(true);
  }
  obs = torch::zeros({rollout_length, num_agents, obs_dim}, opts);
  episode_starts = torch::zeros({rollout_length, num_agents}, opts);
  action_masks = torch::zeros(
      {rollout_length, num_agents, action_dim},
      opts.dtype(torch::kUInt8));
  learner_active = torch::zeros({rollout_length, num_agents}, opts);
  actions = torch::zeros({rollout_length, num_agents}, opts.dtype(torch::kLong));
  action_log_probs = torch::zeros({rollout_length, num_agents}, opts);
  dones = torch::zeros({rollout_length, num_agents}, opts);
  truncated = torch::zeros({rollout_length, num_agents}, opts);
  bootstrap_truncated = torch::zeros({rollout_length, num_agents}, opts);
  goal_positions = torch::zeros({rollout_length, num_agents, 3}, opts);
  terminal_outcome_labels = torch::full(
      {rollout_length, num_agents}, 2,
      opts.dtype(torch::kInt64));
  terminal_observations = torch::zeros({rollout_length, num_agents, obs_dim}, opts);
  mode_ids = torch::zeros({rollout_length, num_agents}, opts.dtype(torch::kInt8));
  encoded_features = torch::zeros({rollout_length, num_agents, encoder_dim}, opts);

  for (const auto& name : head_names) {
    values_[name] = torch::zeros({rollout_length, num_agents}, opts);
    rewards_[name] = torch::zeros({rollout_length, num_agents}, opts);
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
    const torch::Tensor& terminal_outcome_labels_in,
    const torch::Tensor& terminal_observations_in,
    const torch::Tensor& encoded_in) {
  if (step < 0 || step >= rollout_length_) {
    throw std::out_of_range("RolloutStorage::append step is outside rollout capacity.");
  }
  copy_detached(obs[step], obs_in);
  copy_detached(episode_starts[step], episode_starts_in);
  copy_detached(action_masks[step], action_masks_in);
  copy_detached(learner_active[step], learner_active_in);
  copy_detached(actions[step], actions_in);
  copy_detached(action_log_probs[step], action_log_probs_in);
  copy_detached(dones[step], dones_in);
  copy_detached(truncated[step], truncated_in);
  copy_detached(bootstrap_truncated[step], bootstrap_truncated_in);
  copy_detached(goal_positions[step], goal_positions_in);
  copy_detached(terminal_outcome_labels[step], terminal_outcome_labels_in);
  copy_detached(terminal_observations[step], terminal_observations_in);
  if (encoded_in.defined()) {
    copy_detached(encoded_features[step], encoded_in);
  }

  for (const auto& [name, tensor] : values_in) {
    auto it = values_.find(name);
    if (it != values_.end()) {
      copy_detached(it->second[step], tensor);
    }
  }
  for (const auto& [name, tensor] : rewards_in) {
    auto it = rewards_.find(name);
    if (it != rewards_.end()) {
      copy_detached(it->second[step], tensor);
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
    const torch::Tensor& terminal_outcome_labels_in,
    const torch::Tensor& terminal_observations_in,
    const torch::Tensor& encoded_in) {
  if (step < 0 || step >= rollout_length_) {
    throw std::out_of_range("RolloutStorage::append_slice step is outside rollout capacity.");
  }
  const int agent_count = static_cast<int>(obs_in.size(0));
  if (agent_offset < 0 || agent_count < 0 || agent_offset + agent_count > num_agents_) {
    throw std::out_of_range("RolloutStorage::append_slice agent range is outside rollout capacity.");
  }

  copy_detached(obs[step].narrow(0, agent_offset, agent_count), obs_in);
  copy_detached(episode_starts[step].narrow(0, agent_offset, agent_count), episode_starts_in);
  copy_detached(action_masks[step].narrow(0, agent_offset, agent_count), action_masks_in);
  copy_detached(learner_active[step].narrow(0, agent_offset, agent_count), learner_active_in);
  copy_detached(actions[step].narrow(0, agent_offset, agent_count), actions_in);
  copy_detached(action_log_probs[step].narrow(0, agent_offset, agent_count), action_log_probs_in);
  copy_detached(dones[step].narrow(0, agent_offset, agent_count), dones_in);
  copy_detached(truncated[step].narrow(0, agent_offset, agent_count), truncated_in);
  copy_detached(bootstrap_truncated[step].narrow(0, agent_offset, agent_count), bootstrap_truncated_in);
  copy_detached(goal_positions[step].narrow(0, agent_offset, agent_count), goal_positions_in);
  copy_detached(terminal_outcome_labels[step].narrow(0, agent_offset, agent_count), terminal_outcome_labels_in);
  copy_detached(terminal_observations[step].narrow(0, agent_offset, agent_count), terminal_observations_in);
  if (encoded_in.defined()) {
    copy_detached(encoded_features[step].narrow(0, agent_offset, agent_count), encoded_in);
  }

  for (const auto& [name, tensor] : values_in) {
    auto it = values_.find(name);
    if (it != values_.end()) {
      copy_detached(it->second[step].narrow(0, agent_offset, agent_count), tensor);
    }
  }
  for (const auto& [name, tensor] : rewards_in) {
    auto it = rewards_.find(name);
    if (it != rewards_.end()) {
      copy_detached(it->second[step].narrow(0, agent_offset, agent_count), tensor);
    }
  }
}

void RolloutStorage::set_mode_ids_slice(int step, int agent_offset, int agent_count, std::int8_t mode_id) {
  if (step < 0 || step >= rollout_length_) {
    throw std::out_of_range("RolloutStorage::set_mode_ids_slice step is outside rollout capacity.");
  }
  if (agent_offset < 0 || agent_count <= 0 || agent_offset + agent_count > num_agents_) {
    throw std::out_of_range("RolloutStorage::set_mode_ids_slice agent range is outside rollout capacity.");
  }
  mode_ids[step].narrow(0, agent_offset, agent_count).fill_(static_cast<int64_t>(mode_id));
}

void RolloutStorage::set_mode_ids_slice(int step, int agent_offset, std::int8_t mode_id) {
  const int remaining = num_agents_ - agent_offset;
  set_mode_ids_slice(step, agent_offset, remaining, mode_id);
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

void RolloutStorage::mark_step_filled(int step) {
  if (step < 0 || step >= rollout_length_) {
    throw std::out_of_range("RolloutStorage::mark_step_filled step is outside rollout capacity.");
  }
  filled_length_ = std::max(filled_length_, step + 1);
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
      copy_detached(it->second[step], tensor);
    }
  }
}

}  // namespace pulsar

#endif
