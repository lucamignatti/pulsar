#include "pulsar/training/rollout_storage.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

RolloutStorage::RolloutStorage(
    int rollout_length,
    int num_agents,
    int obs_dim,
    torch::Device device)
    : rollout_length_(rollout_length), num_agents_(num_agents) {
  obs = torch::zeros({rollout_length, num_agents, obs_dim}, device);
  episode_starts = torch::zeros({rollout_length, num_agents}, device);
  actions = torch::zeros({rollout_length, num_agents}, torch::TensorOptions().dtype(torch::kLong).device(device));
  log_probs = torch::zeros({rollout_length, num_agents}, device);
  rewards = torch::zeros({rollout_length, num_agents}, device);
  dones = torch::zeros({rollout_length, num_agents}, device);
  sampled_values = torch::zeros({rollout_length, num_agents}, device);
  returns = torch::zeros({rollout_length, num_agents}, device);
  advantages = torch::zeros({rollout_length, num_agents}, device);
}

void RolloutStorage::append(
    int step,
    const torch::Tensor& obs_in,
    const torch::Tensor& episode_starts_in,
    const torch::Tensor& actions_in,
    const torch::Tensor& log_probs_in,
    const torch::Tensor& rewards_in,
    const torch::Tensor& dones_in,
    const torch::Tensor& sampled_values_in) {
  obs[step].copy_(obs_in.detach());
  episode_starts[step].copy_(episode_starts_in.detach());
  actions[step].copy_(actions_in.detach());
  log_probs[step].copy_(log_probs_in.detach());
  rewards[step].copy_(rewards_in.detach());
  dones[step].copy_(dones_in.detach());
  sampled_values[step].copy_(sampled_values_in.detach());
}

void RolloutStorage::compute_returns_and_advantages(
    const torch::Tensor& last_sampled_value,
    float gamma,
    float gae_lambda) {
  torch::Tensor gae = torch::zeros_like(last_sampled_value);
  torch::Tensor next_value = last_sampled_value;

  for (int step = rollout_length_ - 1; step >= 0; --step) {
    const torch::Tensor mask = 1.0 - dones[step];
    const torch::Tensor delta = rewards[step] + gamma * next_value * mask - sampled_values[step];
    gae = delta + gamma * gae_lambda * mask * gae;
    advantages[step] = gae;
    returns[step] = advantages[step] + sampled_values[step];
    next_value = sampled_values[step];
  }
}

int RolloutStorage::rollout_length() const {
  return rollout_length_;
}

int RolloutStorage::num_agents() const {
  return num_agents_;
}

}  // namespace pulsar

#endif
