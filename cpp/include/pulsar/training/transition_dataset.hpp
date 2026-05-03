#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "pulsar/training/offline_dataset.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>
#endif

namespace pulsar {

#ifdef PULSAR_HAS_TORCH

struct TransitionTensorBatch {
  torch::Tensor obs{};
  torch::Tensor actions{};
  torch::Tensor next_obs{};
  torch::Tensor dones{};
};

class TransitionTensorDataset {
 public:
  explicit TransitionTensorDataset(std::string manifest_path, torch::Device device = torch::kCPU, bool allow_pickle = false);

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::int64_t sample_count() const;
  [[nodiscard]] std::int64_t transition_count() const;
  [[nodiscard]] int observation_dim() const;
  [[nodiscard]] int action_dim() const;
  [[nodiscard]] bool has_episode_starts() const;
  [[nodiscard]] bool has_trajectory_end_flags() const;
  [[nodiscard]] const OfflineTensorManifest& manifest() const;
  [[nodiscard]] const torch::Device& device() const;

  void set_device(torch::Device device);

  void for_each_batch(
      int batch_size,
      bool shuffle,
      std::uint64_t seed,
      const std::function<void(const TransitionTensorBatch&)>& fn) const;

 private:
  OfflineTensorManifest manifest_{};
  std::string manifest_path_{};
  std::int64_t transition_count_ = 0;
  torch::Device device_{torch::kCPU};
  bool allow_pickle_ = false;
};

#endif

}  // namespace pulsar
