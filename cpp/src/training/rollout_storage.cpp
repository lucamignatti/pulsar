#include "pulsar/training/rollout_storage.hpp"

#ifdef PULSAR_HAS_TORCH

#include "pulsar/training/lfpo_math.hpp"

namespace pulsar {

RolloutStorage::RolloutStorage(
    int rollout_length,
    int num_agents,
    int obs_dim,
    int action_dim,
    int candidate_count,
    torch::Device device)
    : rollout_length_(rollout_length),
      num_agents_(num_agents),
      candidate_count_(candidate_count),
      device_(device) {
  raw_obs = torch::zeros({rollout_length, num_agents, obs_dim}, device);
  final_raw_obs = torch::zeros({num_agents, obs_dim}, device);
  obs = torch::zeros({rollout_length, num_agents, obs_dim}, device);
  episode_starts = torch::zeros({rollout_length, num_agents}, device);
  action_masks = torch::zeros(
      {rollout_length, num_agents, action_dim},
      torch::TensorOptions().dtype(torch::kUInt8).device(device));
  learner_active = torch::zeros({rollout_length, num_agents}, device);
  executed_actions = torch::zeros({rollout_length, num_agents}, torch::TensorOptions().dtype(torch::kLong).device(device));
  candidate_actions = torch::zeros(
      {rollout_length, num_agents, candidate_count},
      torch::TensorOptions().dtype(torch::kLong).device(device));
  candidate_log_probs = torch::zeros({rollout_length, num_agents, candidate_count}, device);
  trajectory_ids = torch::zeros({rollout_length, num_agents}, torch::TensorOptions().dtype(torch::kLong).device(device));
  dones = torch::zeros({rollout_length, num_agents}, device);
  terminal_outcomes = torch::full(
      {rollout_length, num_agents},
      2,
      torch::TensorOptions().dtype(torch::kLong).device(device));
  terminal_raw_obs = torch::zeros({rollout_length, num_agents, obs_dim}, device);
}

void RolloutStorage::append(
    int step,
    const torch::Tensor& raw_obs_in,
    const torch::Tensor& obs_in,
    const torch::Tensor& episode_starts_in,
    const torch::Tensor& action_masks_in,
    const torch::Tensor& learner_active_in,
    const torch::Tensor& executed_actions_in,
    const torch::Tensor& candidate_actions_in,
    const torch::Tensor& candidate_log_probs_in,
    const torch::Tensor& trajectory_ids_in,
    const torch::Tensor& dones_in,
    const torch::Tensor& terminal_outcomes_in,
    const torch::Tensor& terminal_raw_obs_in) {
  raw_obs[step].copy_(raw_obs_in.detach());
  obs[step].copy_(obs_in.detach());
  episode_starts[step].copy_(episode_starts_in.detach());
  action_masks[step].copy_(action_masks_in.detach());
  learner_active[step].copy_(learner_active_in.detach());
  executed_actions[step].copy_(executed_actions_in.detach());
  candidate_actions[step].copy_(candidate_actions_in.detach());
  candidate_log_probs[step].copy_(candidate_log_probs_in.detach());
  trajectory_ids[step].copy_(trajectory_ids_in.detach());
  dones[step].copy_(dones_in.detach());
  terminal_outcomes[step].copy_(terminal_outcomes_in.detach());
  terminal_raw_obs[step].copy_(terminal_raw_obs_in.detach());
}

void RolloutStorage::set_final_observation(const torch::Tensor& raw_obs_in) {
  final_raw_obs.copy_(raw_obs_in.detach());
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

int RolloutStorage::candidate_count() const {
  return candidate_count_;
}

}  // namespace pulsar

#endif
