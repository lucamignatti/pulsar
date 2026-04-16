#pragma once

#ifdef PULSAR_HAS_TORCH

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"

namespace pulsar {

struct NGPTrajectory {
  torch::Tensor obs_cpu{};
  std::int64_t label = 0;
};

std::int64_t ngp_trajectory_sample_count(const std::vector<NGPTrajectory>& trajectories);
std::vector<NGPTrajectory> load_ngp_trajectories_from_manifest(const std::string& manifest_path);
std::vector<NGPTrajectory> select_ngp_trajectory_subset(
    const std::vector<NGPTrajectory>& trajectories,
    std::int64_t target_samples,
    std::uint64_t seed);

class OnlineNGPReplayBuffer {
 public:
  struct AgentTrajectory {
    std::vector<float> obs{};
    std::size_t steps = 0;
  };

  OnlineNGPReplayBuffer(
      RewardConfig::OnlineDatasetExportConfig export_config,
      RewardConfig::RefreshConfig refresh_config,
      int obs_dim,
      std::size_t num_envs,
      std::size_t agents_per_env);

  void record_step(
      const torch::Tensor& raw_obs_cpu,
      const torch::Tensor& dones_cpu,
      const torch::Tensor& terminated_cpu,
      const torch::Tensor& truncated_cpu,
      const torch::Tensor& terminal_next_goal_labels_cpu);
  void close_window();
  void clear_completed_windows();

  [[nodiscard]] std::vector<NGPTrajectory> train_trajectories() const;
  [[nodiscard]] std::vector<NGPTrajectory> val_trajectories() const;
  [[nodiscard]] std::int64_t train_sample_count() const;
  [[nodiscard]] std::int64_t val_sample_count() const;
  [[nodiscard]] std::int64_t trajectories_written() const;
  [[nodiscard]] std::size_t retained_window_count() const;
  [[nodiscard]] std::shared_ptr<OnlineNGPReplayBuffer> clone() const;

  void save(const std::filesystem::path& directory) const;
  void load(const std::filesystem::path& directory);

 private:
  struct Window {
    std::int64_t id = 0;
    std::vector<NGPTrajectory> train{};
    std::vector<NGPTrajectory> val{};
    std::int64_t train_samples = 0;
    std::int64_t val_samples = 0;
  };

  [[nodiscard]] std::string choose_split() const;
  void enforce_retention_limits();
  void save_window(const Window& window, const std::filesystem::path& directory, const std::string& prefix) const;
  void load_window(Window* window, const std::filesystem::path& directory, const std::string& prefix) const;

  RewardConfig::OnlineDatasetExportConfig export_config_{};
  RewardConfig::RefreshConfig refresh_config_{};
  int obs_dim_ = 0;
  std::size_t num_envs_ = 0;
  std::size_t agents_per_env_ = 0;
  std::vector<AgentTrajectory> trajectories_{};
  std::vector<std::uint64_t> env_episode_ids_{};
  std::uint64_t completed_episodes_ = 0;
  std::deque<Window> windows_{};
  Window current_window_{};
  std::int64_t next_window_id_ = 0;
  std::int64_t trajectories_written_ = 0;
};

}  // namespace pulsar

#endif
