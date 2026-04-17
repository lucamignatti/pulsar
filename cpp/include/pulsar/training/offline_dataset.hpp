#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "pulsar/config/config.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>
#endif

namespace pulsar {

#ifdef PULSAR_HAS_TORCH

struct OfflineTensorBatch {
  torch::Tensor obs{};
  torch::Tensor actions{};
  torch::Tensor action_probs{};
  torch::Tensor next_goal{};
  torch::Tensor weights{};
  torch::Tensor episode_starts{};
  torch::Tensor terminated{};
  torch::Tensor truncated{};
};

struct OfflineTensorPackedBatch {
  torch::Tensor obs{};
  torch::Tensor actions{};
  torch::Tensor action_probs{};
  torch::Tensor next_goal{};
  torch::Tensor weights{};
  torch::Tensor episode_starts{};
  torch::Tensor terminated{};
  torch::Tensor truncated{};
  torch::Tensor valid_mask{};
  std::vector<std::int64_t> lengths{};
};

struct OfflineTensorShardEntry {
  std::string obs_path{};
  std::string actions_path{};
  std::string action_probs_path{};
  std::string next_goal_path{};
  std::string weights_path{};
  std::string episode_starts_path{};
  std::string terminated_path{};
  std::string truncated_path{};
  std::int64_t samples = 0;
};

struct OfflineTensorManifest {
  int schema_version = 1;
  int observation_dim = 0;
  int action_dim = 0;
  int next_goal_classes = 3;
  std::vector<OfflineTensorShardEntry> shards{};
};

class OfflineTensorDataset {
 public:
  explicit OfflineTensorDataset(std::string manifest_path);

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::int64_t sample_count() const;
  [[nodiscard]] int observation_dim() const;
  [[nodiscard]] int action_dim() const;
  [[nodiscard]] int next_goal_classes() const;
  [[nodiscard]] bool has_episode_starts() const;
  [[nodiscard]] bool has_trajectory_end_flags() const;
  [[nodiscard]] const OfflineTensorManifest& manifest() const;

  void for_each_batch(
      int batch_size,
      bool shuffle,
      std::uint64_t seed,
      const std::function<void(const OfflineTensorBatch&)>& fn) const;

  void for_each_trajectory(
      bool shuffle,
      std::uint64_t seed,
      const std::function<void(const OfflineTensorBatch&)>& fn) const;

  void for_each_packed_trajectory_batch(
      int max_tokens,
      bool shuffle,
      std::uint64_t seed,
      const std::function<void(const OfflineTensorPackedBatch&)>& fn) const;

 private:
  OfflineTensorManifest manifest_{};
  std::string manifest_path_{};
  std::int64_t sample_count_ = 0;
};

OfflineTensorManifest load_offline_tensor_manifest(const std::string& path);

#endif

}  // namespace pulsar
