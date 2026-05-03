#include "pulsar/training/rollout_storage.hpp"

#ifdef PULSAR_HAS_TORCH

#include "pulsar/training/ppo_math.hpp"

namespace pulsar {

RolloutStorage::RolloutStorage(
    int rollout_length,
    int num_agents,
    int obs_dim,
    int action_dim,
    torch::Device device)
    : rollout_length_(rollout_length),
      num_agents_(num_agents),
      device_(device) {
  raw_obs = torch::zeros({rollout_length, num_agents, obs_dim}, device);
  final_raw_obs = torch::zeros({num_agents, obs_dim}, device);
  obs = torch::zeros({rollout_length, num_agents, obs_dim}, device);
  episode_starts = torch::zeros({rollout_length, num_agents}, device);
  action_masks = torch::zeros(
      {rollout_length, num_agents, action_dim},
      torch::TensorOptions().dtype(torch::kUInt8).device(device));
  learner_active = torch::zeros({rollout_length, num_agents}, device);
  actions = torch::zeros({rollout_length, num_agents}, torch::TensorOptions().dtype(torch::kLong).device(device));
  action_log_probs = torch::zeros({rollout_length, num_agents}, device);
  dones = torch::zeros({rollout_length, num_agents}, device);

  const std::vector<std::string> head_names = {"extrinsic", "curiosity", "learning_progress", "controllability"};
  for (const auto& name : head_names) {
    values_[name] = torch::zeros({rollout_length, num_agents}, device);
    rewards_[name] = torch::zeros({rollout_length, num_agents}, device);
  }
}

void RolloutStorage::append(
    int step,
    const torch::Tensor& raw_obs_in,
    const torch::Tensor& obs_in,
    const torch::Tensor& episode_starts_in,
    const torch::Tensor& action_masks_in,
    const torch::Tensor& learner_active_in,
    const torch::Tensor& actions_in,
    const torch::Tensor& action_log_probs_in,
    const std::unordered_map<std::string, torch::Tensor>& values_in,
    const std::unordered_map<std::string, torch::Tensor>& rewards_in,
    const torch::Tensor& dones_in) {
  raw_obs[step].copy_(raw_obs_in.detach());
  obs[step].copy_(obs_in.detach());
  episode_starts[step].copy_(episode_starts_in.detach());
  action_masks[step].copy_(action_masks_in.detach());
  learner_active[step].copy_(learner_active_in.detach());
  actions[step].copy_(actions_in.detach());
  action_log_probs[step].copy_(action_log_probs_in.detach());
  dones[step].copy_(dones_in.detach());

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
}

void RolloutStorage::set_final_observation(const torch::Tensor& raw_obs_in) {
  final_raw_obs.copy_(raw_obs_in.detach());
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

void RolloutStorage::set_initial_state(const ContinuumState& state) {
  initial_state = state_to_device(clone_state(state), device_);
}

ContinuumState RolloutStorage::initial_state_for_agents(const torch::Tensor& agent_indices) const {
  return gather_state(initial_state, agent_indices);
}

int RolloutStorage::rollout_length() const {
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

}  // namespace pulsar

#endif