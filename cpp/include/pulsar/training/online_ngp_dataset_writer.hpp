#pragma once

#ifdef PULSAR_HAS_TORCH

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"

namespace pulsar {

class OnlineNGPDatasetWriter {
 public:
  OnlineNGPDatasetWriter(
      RewardConfig::OnlineDatasetExportConfig config,
      std::filesystem::path output_root,
      int obs_dim,
      int action_dim,
      std::size_t num_envs,
      std::size_t agents_per_env);

  void record_step(
      const torch::Tensor& raw_obs_cpu,
      const torch::Tensor& dones_cpu,
      const torch::Tensor& terminated_cpu,
      const torch::Tensor& truncated_cpu,
      const torch::Tensor& terminal_next_goal_labels_cpu);
  void flush_pending();
  void finish();

  [[nodiscard]] std::int64_t samples_written() const;
  [[nodiscard]] std::int64_t trajectories_written() const;
  [[nodiscard]] const std::filesystem::path& output_root() const;

 private:
  struct AgentTrajectory {
    std::vector<float> obs{};
    std::size_t steps = 0;
  };

  struct SplitBuffers {
    std::vector<float> obs{};
    std::vector<std::int64_t> next_goal{};
    std::vector<float> weights{};
    std::vector<float> episode_starts{};
    std::vector<float> terminated{};
    std::vector<float> truncated{};
    std::vector<std::string> shards{};
    std::int64_t shard_index = 0;
  };

  [[nodiscard]] std::string choose_split(std::size_t env_idx) const;
  void flush_ready_split(const std::string& split);
  void flush_split(SplitBuffers& buffers, const std::string& split);
  void prune_old_shards(const SplitBuffers& buffers, const std::filesystem::path& split_dir) const;
  void write_manifest(const SplitBuffers& buffers, const std::filesystem::path& path) const;

  RewardConfig::OnlineDatasetExportConfig config_{};
  std::filesystem::path output_root_{};
  int obs_dim_ = 0;
  int action_dim_ = 0;
  std::size_t num_envs_ = 0;
  std::size_t agents_per_env_ = 0;
  std::vector<AgentTrajectory> trajectories_{};
  std::vector<std::uint64_t> env_episode_ids_{};
  SplitBuffers train_;
  SplitBuffers val_;
  std::int64_t samples_written_ = 0;
  std::int64_t trajectories_written_ = 0;
};

}  // namespace pulsar

#endif
