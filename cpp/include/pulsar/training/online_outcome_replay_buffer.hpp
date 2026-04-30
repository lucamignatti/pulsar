#pragma once

#ifdef PULSAR_HAS_TORCH

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"

namespace pulsar {

struct OutcomeTrajectory {
  torch::Tensor obs_cpu{};
  std::int64_t outcome = 2;
};

std::int64_t outcome_trajectory_sample_count(const std::vector<OutcomeTrajectory>& trajectories);

class OnlineOutcomeReplayBuffer {
 public:
  struct AgentTrajectory {
    std::vector<float> obs{};
    std::size_t steps = 0;
  };

  OnlineOutcomeReplayBuffer(
      int obs_dim,
      std::size_t num_envs,
      std::size_t agents_per_env,
      int retained_trajectories);

  void record_step(
      const torch::Tensor& raw_obs_cpu,
      const torch::Tensor& dones_cpu,
      const torch::Tensor& terminal_outcome_labels_cpu,
      const torch::Tensor& terminal_obs_cpu);
  void clear();

  [[nodiscard]] std::vector<OutcomeTrajectory> trajectories() const;
  [[nodiscard]] std::int64_t sample_count() const;
  [[nodiscard]] std::int64_t trajectories_written() const;
  [[nodiscard]] std::shared_ptr<OnlineOutcomeReplayBuffer> clone() const;

 private:
  void enforce_retention();

  int obs_dim_ = 0;
  std::size_t num_envs_ = 0;
  std::size_t agents_per_env_ = 0;
  int retained_trajectories_ = 0;
  std::vector<AgentTrajectory> partial_{};
  std::deque<OutcomeTrajectory> completed_{};
  std::int64_t trajectories_written_ = 0;
};

}  // namespace pulsar

#endif
