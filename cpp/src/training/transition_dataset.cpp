#include "pulsar/training/transition_dataset.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/serialize.h>

namespace pulsar {
namespace {

constexpr float kBoundaryThreshold = 0.5F;

torch::Tensor load_tensor_checked(const std::string& path, bool allow_pickle = false) {
  try {
    torch::Tensor tensor;
    torch::load(tensor, path);
    if (!tensor.defined()) {
      throw std::runtime_error("Loaded undefined tensor.");
    }
    return tensor.contiguous();
  } catch (const c10::Error&) {
    if (!allow_pickle) {
      throw std::runtime_error(
          "Tensor load failed for '" + path +
          "' and pickle fallback is disabled. "
          "Set OfflineDatasetConfig.allow_pickle=true if the shard files are trusted.");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Failed to open tensor file: " + path);
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
      throw std::runtime_error("Tensor file was empty: " + path);
    }
    const torch::IValue value = torch::pickle_load(bytes);
    if (!value.isTensor()) {
      throw std::runtime_error("Tensor file did not contain a tensor: " + path);
    }
    return value.toTensor().contiguous();
  }
}

struct LoadedTransitionShard {
  torch::Tensor obs{};
  torch::Tensor actions{};
  torch::Tensor episode_starts{};
  torch::Tensor terminated{};
  torch::Tensor truncated{};
};

LoadedTransitionShard load_shard_tensors(
    const std::filesystem::path& manifest_dir,
    const OfflineTensorShardEntry& shard,
    int observation_dim,
    bool allow_pickle = false) {
  LoadedTransitionShard loaded;
  loaded.obs = load_tensor_checked((manifest_dir / shard.obs_path).string(), allow_pickle).to(torch::kFloat32).contiguous();
  if (loaded.obs.dim() != 2 || loaded.obs.size(1) != observation_dim) {
    throw std::runtime_error("Transition shard obs tensor has unexpected shape.");
  }

  if (shard.actions_path.empty()) {
    throw std::runtime_error("TransitionTensorDataset requires actions_path in every shard.");
  }
  loaded.actions = load_tensor_checked((manifest_dir / shard.actions_path).string(), allow_pickle).to(torch::kLong).view({-1});

  if (!shard.episode_starts_path.empty()) {
    loaded.episode_starts =
        load_tensor_checked((manifest_dir / shard.episode_starts_path).string(), allow_pickle).to(torch::kFloat32).view({-1});
  } else {
    loaded.episode_starts = torch::zeros({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
    if (loaded.obs.size(0) > 0) {
      loaded.episode_starts[0] = 1.0F;
    }
  }
  loaded.terminated = !shard.terminated_path.empty()
      ? load_tensor_checked((manifest_dir / shard.terminated_path).string(), allow_pickle).to(torch::kFloat32).view({-1})
      : torch::zeros({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
  loaded.truncated = !shard.truncated_path.empty()
      ? load_tensor_checked((manifest_dir / shard.truncated_path).string(), allow_pickle).to(torch::kFloat32).view({-1})
      : torch::zeros({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));

  if (loaded.obs.size(0) != loaded.actions.size(0) ||
      loaded.obs.size(0) != loaded.episode_starts.size(0) ||
      loaded.obs.size(0) != loaded.terminated.size(0) ||
      loaded.obs.size(0) != loaded.truncated.size(0)) {
    throw std::runtime_error("Transition shard tensors have mismatched leading dimensions.");
  }
  return loaded;
}

torch::Tensor build_done_flags(const LoadedTransitionShard& loaded) {
  const std::int64_t transition_count = std::max<std::int64_t>(loaded.obs.size(0) - 1, 0);
  torch::Tensor done_flags = torch::zeros({transition_count}, torch::TensorOptions().dtype(torch::kBool));
  if (transition_count == 0) {
    return done_flags;
  }

  // The offline frame format does not carry an explicit next_obs tensor. We keep
  // consecutive frame pairs inside each shard and mark boundary-crossing pairs via
  // dones so downstream callers can mask them if needed.
  const torch::Tensor episode_cpu = loaded.episode_starts.to(torch::kCPU).contiguous();
  const torch::Tensor terminated_cpu = loaded.terminated.to(torch::kCPU).contiguous();
  const torch::Tensor truncated_cpu = loaded.truncated.to(torch::kCPU).contiguous();
  const float* episode_ptr = episode_cpu.data_ptr<float>();
  const float* terminated_ptr = terminated_cpu.data_ptr<float>();
  const float* truncated_ptr = truncated_cpu.data_ptr<float>();
  bool* done_ptr = done_flags.data_ptr<bool>();
  for (std::int64_t i = 0; i < transition_count; ++i) {
    done_ptr[i] = terminated_ptr[i] > kBoundaryThreshold || truncated_ptr[i] > kBoundaryThreshold ||
        episode_ptr[i + 1] > kBoundaryThreshold;
  }
  return done_flags;
}

}  // namespace

TransitionTensorDataset::TransitionTensorDataset(std::string manifest_path, torch::Device device, bool allow_pickle)
    : manifest_(load_offline_tensor_manifest(manifest_path)),
      manifest_path_(std::move(manifest_path)),
      device_(std::move(device)),
      allow_pickle_(allow_pickle) {
  for (const auto& shard : manifest_.shards) {
    if (shard.actions_path.empty()) {
      throw std::runtime_error("TransitionTensorDataset requires actions_path in every shard.");
    }
    if (shard.samples > 1) {
      transition_count_ += shard.samples - 1;
    }
  }
}

bool TransitionTensorDataset::empty() const {
  return manifest_.shards.empty() || transition_count_ == 0;
}

std::int64_t TransitionTensorDataset::sample_count() const {
  return transition_count_;
}

std::int64_t TransitionTensorDataset::transition_count() const {
  return transition_count_;
}

int TransitionTensorDataset::observation_dim() const {
  return manifest_.observation_dim;
}

int TransitionTensorDataset::action_dim() const {
  return manifest_.action_dim;
}

bool TransitionTensorDataset::has_episode_starts() const {
  for (const auto& shard : manifest_.shards) {
    if (shard.episode_starts_path.empty()) {
      return false;
    }
  }
  return !manifest_.shards.empty();
}

bool TransitionTensorDataset::has_trajectory_end_flags() const {
  for (const auto& shard : manifest_.shards) {
    if (shard.terminated_path.empty() || shard.truncated_path.empty()) {
      return false;
    }
  }
  return !manifest_.shards.empty();
}

const OfflineTensorManifest& TransitionTensorDataset::manifest() const {
  return manifest_;
}

const torch::Device& TransitionTensorDataset::device() const {
  return device_;
}

void TransitionTensorDataset::set_device(torch::Device device) {
  device_ = std::move(device);
}

void TransitionTensorDataset::for_each_batch(
    int batch_size,
    bool shuffle,
    std::uint64_t seed,
    const std::function<void(const TransitionTensorBatch&)>& fn) const {
  if (batch_size <= 0) {
    throw std::invalid_argument("TransitionTensorDataset batch size must be positive.");
  }

  const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path_).parent_path();
  std::mt19937_64 rng(seed);
  for (const auto& shard : manifest_.shards) {
    const LoadedTransitionShard loaded = load_shard_tensors(manifest_dir, shard, manifest_.observation_dim, allow_pickle_);
    const std::int64_t shard_transition_count = std::max<std::int64_t>(loaded.obs.size(0) - 1, 0);
    if (shard_transition_count == 0) {
      continue;
    }

    const torch::Tensor done_flags = build_done_flags(loaded);
    std::vector<std::int64_t> ordering(static_cast<std::size_t>(shard_transition_count));
    std::iota(ordering.begin(), ordering.end(), 0);
    if (shuffle) {
      std::shuffle(ordering.begin(), ordering.end(), rng);
    }

    for (std::int64_t offset = 0; offset < shard_transition_count; offset += batch_size) {
      const std::int64_t length = std::min<std::int64_t>(batch_size, shard_transition_count - offset);
      torch::Tensor batch_indices;
      if (shuffle) {
        const auto begin = ordering.begin() + static_cast<std::ptrdiff_t>(offset);
        const auto end = begin + static_cast<std::ptrdiff_t>(length);
        batch_indices = torch::tensor(
            std::vector<std::int64_t>(begin, end),
            torch::TensorOptions().dtype(torch::kLong));
      } else {
        batch_indices = torch::arange(offset, offset + length, torch::TensorOptions().dtype(torch::kLong));
      }

      TransitionTensorBatch batch;
      batch.obs = loaded.obs.index_select(0, batch_indices).to(device_);
      batch.actions = loaded.actions.index_select(0, batch_indices).to(device_);
      batch.next_obs = loaded.obs.index_select(0, batch_indices + 1).to(device_);
      batch.dones = done_flags.index_select(0, batch_indices).to(device_);
      fn(batch);
    }
  }
}

}  // namespace pulsar

#endif
