#include "pulsar/training/online_outcome_replay_buffer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <stdexcept>

namespace pulsar {

std::int64_t outcome_trajectory_sample_count(const std::vector<OutcomeTrajectory>& trajectories) {
  std::int64_t samples = 0;
  for (const auto& trajectory : trajectories) {
    samples += trajectory.obs_cpu.defined() ? trajectory.obs_cpu.size(0) : 0;
  }
  return samples;
}

OnlineOutcomeReplayBuffer::OnlineOutcomeReplayBuffer(
    int obs_dim,
    std::size_t num_envs,
    std::size_t agents_per_env,
    int retained_trajectories)
    : obs_dim_(obs_dim),
      num_envs_(num_envs),
      agents_per_env_(agents_per_env),
      retained_trajectories_(retained_trajectories),
      partial_(num_envs * agents_per_env) {
  if (obs_dim_ <= 0 || num_envs_ == 0 || agents_per_env_ == 0) {
    throw std::invalid_argument("OnlineOutcomeReplayBuffer requires positive dimensions.");
  }
}

void OnlineOutcomeReplayBuffer::record_step(
    const torch::Tensor& raw_obs_cpu,
    const torch::Tensor& dones_cpu,
    const torch::Tensor& terminal_outcome_labels_cpu,
    const torch::Tensor& terminal_obs_cpu) {
  if (raw_obs_cpu.device().type() != torch::kCPU ||
      dones_cpu.device().type() != torch::kCPU ||
      terminal_outcome_labels_cpu.device().type() != torch::kCPU ||
      terminal_obs_cpu.device().type() != torch::kCPU) {
    throw std::invalid_argument("OnlineOutcomeReplayBuffer expects CPU tensors.");
  }
  const std::size_t total_agents = num_envs_ * agents_per_env_;
  if (static_cast<std::size_t>(raw_obs_cpu.size(0)) != total_agents ||
      static_cast<std::size_t>(dones_cpu.numel()) != total_agents ||
      static_cast<std::size_t>(terminal_outcome_labels_cpu.numel()) != total_agents ||
      static_cast<std::size_t>(terminal_obs_cpu.size(0)) != total_agents ||
      terminal_obs_cpu.size(1) != obs_dim_) {
    throw std::invalid_argument("OnlineOutcomeReplayBuffer tensor sizes disagree.");
  }
  const torch::Tensor obs = raw_obs_cpu.contiguous();
  const torch::Tensor dones = dones_cpu.contiguous();
  const torch::Tensor labels = terminal_outcome_labels_cpu.contiguous();
  const torch::Tensor terminal_obs = terminal_obs_cpu.contiguous();
  const float* obs_ptr = obs.data_ptr<float>();
  const float* dones_ptr = dones.data_ptr<float>();
  const std::int64_t* labels_ptr = labels.data_ptr<std::int64_t>();
  const float* terminal_obs_ptr = terminal_obs.data_ptr<float>();

  for (std::size_t env_idx = 0; env_idx < num_envs_; ++env_idx) {
    const std::size_t env_base = env_idx * agents_per_env_;
    for (std::size_t local_idx = 0; local_idx < agents_per_env_; ++local_idx) {
      const std::size_t agent_idx = env_base + local_idx;
      AgentTrajectory& trajectory = partial_[agent_idx];
      const float* row = obs_ptr + static_cast<std::ptrdiff_t>(agent_idx * static_cast<std::size_t>(obs_dim_));
      trajectory.obs.insert(trajectory.obs.end(), row, row + obs_dim_);
      trajectory.steps += 1;
    }

    if (dones_ptr[env_base] <= 0.5F) {
      continue;
    }

    for (std::size_t local_idx = 0; local_idx < agents_per_env_; ++local_idx) {
      const std::size_t agent_idx = env_base + local_idx;
      AgentTrajectory& partial = partial_[agent_idx];
      if (partial.steps == 0) {
        continue;
      }
      const float* terminal_row =
          terminal_obs_ptr + static_cast<std::ptrdiff_t>(agent_idx * static_cast<std::size_t>(obs_dim_));
      partial.obs.insert(partial.obs.end(), terminal_row, terminal_row + obs_dim_);
      partial.steps += 1;
      OutcomeTrajectory completed;
      completed.obs_cpu =
          torch::from_blob(
              partial.obs.data(),
              {static_cast<long>(partial.steps), static_cast<long>(obs_dim_)},
              torch::TensorOptions().dtype(torch::kFloat32))
              .clone();
      completed.outcome = labels_ptr[agent_idx];
      completed_.push_back(std::move(completed));
      partial.obs.clear();
      partial.steps = 0;
      trajectories_written_ += 1;
    }
    enforce_retention();
  }
}

void OnlineOutcomeReplayBuffer::clear() {
  completed_.clear();
}

std::vector<OutcomeTrajectory> OnlineOutcomeReplayBuffer::trajectories() const {
  return std::vector<OutcomeTrajectory>(completed_.begin(), completed_.end());
}

std::int64_t OnlineOutcomeReplayBuffer::sample_count() const {
  std::int64_t samples = 0;
  for (const auto& trajectory : completed_) {
    samples += trajectory.obs_cpu.defined() ? trajectory.obs_cpu.size(0) : 0;
  }
  return samples;
}

std::int64_t OnlineOutcomeReplayBuffer::trajectories_written() const {
  return trajectories_written_;
}

std::shared_ptr<OnlineOutcomeReplayBuffer> OnlineOutcomeReplayBuffer::clone() const {
  return std::make_shared<OnlineOutcomeReplayBuffer>(*this);
}

void OnlineOutcomeReplayBuffer::enforce_retention() {
  if (retained_trajectories_ <= 0) {
    return;
  }
  while (static_cast<int>(completed_.size()) > retained_trajectories_) {
    completed_.pop_front();
  }
}

}  // namespace pulsar

#endif
