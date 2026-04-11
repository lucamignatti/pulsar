#include "pulsar/training/rollout_storage.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

RolloutStorage::RolloutStorage(
    int rollout_length,
    int num_envs,
    int obs_dim,
    int,
    torch::Device device)
    : rollout_length_(rollout_length), num_envs_(num_envs) {
  obs = torch::zeros({rollout_length, num_envs, obs_dim}, device);
  actions = torch::zeros({rollout_length, num_envs}, torch::TensorOptions().dtype(torch::kLong).device(device));
  log_probs = torch::zeros({rollout_length, num_envs}, device);
  rewards = torch::zeros({rollout_length, num_envs}, device);
  dones = torch::zeros({rollout_length, num_envs}, device);
  values = torch::zeros({rollout_length, num_envs}, device);
  returns = torch::zeros({rollout_length, num_envs}, device);
  advantages = torch::zeros({rollout_length, num_envs}, device);
}

void RolloutStorage::append(
    int step,
    const torch::Tensor& obs_in,
    const torch::Tensor& actions_in,
    const torch::Tensor& log_probs_in,
    const torch::Tensor& rewards_in,
    const torch::Tensor& dones_in,
    const torch::Tensor& values_in) {
  obs[step].copy_(obs_in.detach());
  actions[step].copy_(actions_in.detach());
  log_probs[step].copy_(log_probs_in.detach());
  rewards[step].copy_(rewards_in.detach());
  dones[step].copy_(dones_in.detach());
  values[step].copy_(values_in.detach());
}

void RolloutStorage::compute_returns_and_advantages(
    const torch::Tensor& last_value,
    float gamma,
    float gae_lambda) {
  torch::Tensor gae = torch::zeros_like(last_value);
  torch::Tensor next_value = last_value;

  for (int step = rollout_length_ - 1; step >= 0; --step) {
    const torch::Tensor mask = 1.0 - dones[step];
    const torch::Tensor delta = rewards[step] + gamma * next_value * mask - values[step];
    gae = delta + gamma * gae_lambda * mask * gae;
    advantages[step] = gae;
    returns[step] = advantages[step] + values[step];
    next_value = values[step];
  }
}

int RolloutStorage::rollout_length() const {
  return rollout_length_;
}

int RolloutStorage::num_envs() const {
  return num_envs_;
}

}  // namespace pulsar

#endif
