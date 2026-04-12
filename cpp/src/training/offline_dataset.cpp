#include "pulsar/training/offline_dataset.hpp"

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <torch/serialize.h>

namespace pulsar {
namespace {

using nlohmann::json;

torch::Tensor load_tensor_checked(const std::string& path) {
  try {
    torch::Tensor tensor;
    torch::load(tensor, path);
    if (!tensor.defined()) {
      throw std::runtime_error("Loaded undefined tensor.");
    }
    return tensor.contiguous();
  } catch (const c10::Error&) {
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

}  // namespace

OfflineTensorManifest load_offline_tensor_manifest(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open offline manifest: " + path);
  }

  json j;
  input >> j;

  OfflineTensorManifest manifest;
  manifest.schema_version = j.value("schema_version", 1);
  manifest.observation_dim = j.at("observation_dim").get<int>();
  manifest.action_dim = j.at("action_dim").get<int>();
  manifest.next_goal_classes = j.value("next_goal_classes", 3);
  for (const auto& shard_json : j.at("shards")) {
    OfflineTensorShardEntry shard;
    shard.obs_path = shard_json.at("obs_path").get<std::string>();
    shard.actions_path = shard_json.value("actions_path", std::string{});
    shard.action_probs_path = shard_json.value("action_probs_path", std::string{});
    shard.next_goal_path = shard_json.value("next_goal_path", std::string{});
    shard.weights_path = shard_json.value("weights_path", std::string{});
    shard.episode_starts_path = shard_json.value("episode_starts_path", std::string{});
    shard.samples = shard_json.at("samples").get<std::int64_t>();
    manifest.shards.push_back(std::move(shard));
  }
  return manifest;
}

OfflineTensorDataset::OfflineTensorDataset(std::string manifest_path)
    : manifest_(load_offline_tensor_manifest(manifest_path)), manifest_path_(std::move(manifest_path)) {
  for (const auto& shard : manifest_.shards) {
    sample_count_ += shard.samples;
  }
}

bool OfflineTensorDataset::empty() const {
  return manifest_.shards.empty() || sample_count_ == 0;
}

std::int64_t OfflineTensorDataset::sample_count() const {
  return sample_count_;
}

int OfflineTensorDataset::observation_dim() const {
  return manifest_.observation_dim;
}

int OfflineTensorDataset::action_dim() const {
  return manifest_.action_dim;
}

int OfflineTensorDataset::next_goal_classes() const {
  return manifest_.next_goal_classes;
}

bool OfflineTensorDataset::has_episode_starts() const {
  for (const auto& shard : manifest_.shards) {
    if (shard.episode_starts_path.empty()) {
      return false;
    }
  }
  return !manifest_.shards.empty();
}

const OfflineTensorManifest& OfflineTensorDataset::manifest() const {
  return manifest_;
}

void OfflineTensorDataset::for_each_batch(
    int batch_size,
    bool shuffle,
    std::uint64_t seed,
    const std::function<void(const OfflineTensorBatch&)>& fn) const {
  if (batch_size <= 0) {
    throw std::invalid_argument("OfflineTensorDataset batch size must be positive.");
  }

  const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path_).parent_path();
  std::uint64_t shard_seed = seed;
  for (const auto& shard : manifest_.shards) {
    torch::Tensor obs = load_tensor_checked((manifest_dir / shard.obs_path).string());
    if (obs.dim() != 2 || obs.size(1) != manifest_.observation_dim) {
      throw std::runtime_error("Offline shard obs tensor has unexpected shape.");
    }

    torch::Tensor actions;
    if (!shard.actions_path.empty()) {
      actions = load_tensor_checked((manifest_dir / shard.actions_path).string()).to(torch::kLong).view({-1});
    }

    torch::Tensor action_probs;
    if (!shard.action_probs_path.empty()) {
      action_probs =
          load_tensor_checked((manifest_dir / shard.action_probs_path).string()).to(torch::kFloat32).contiguous();
    }

    torch::Tensor next_goal;
    if (!shard.next_goal_path.empty()) {
      next_goal = load_tensor_checked((manifest_dir / shard.next_goal_path).string()).to(torch::kLong).view({-1});
    }

    torch::Tensor weights;
    if (!shard.weights_path.empty()) {
      weights = load_tensor_checked((manifest_dir / shard.weights_path).string()).to(torch::kFloat32).view({-1});
    } else {
      weights = torch::ones({obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
    }

    torch::Tensor episode_starts;
    if (!shard.episode_starts_path.empty()) {
      episode_starts =
          load_tensor_checked((manifest_dir / shard.episode_starts_path).string()).to(torch::kFloat32).view({-1});
    } else {
      episode_starts = torch::zeros({obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
      if (obs.size(0) > 0) {
        episode_starts[0] = 1.0F;
      }
    }

    if (obs.size(0) != weights.size(0) ||
        (actions.defined() && obs.size(0) != actions.size(0)) ||
        (action_probs.defined() && (action_probs.dim() != 2 || obs.size(0) != action_probs.size(0))) ||
        (next_goal.defined() && obs.size(0) != next_goal.size(0)) ||
        obs.size(0) != episode_starts.size(0)) {
      throw std::runtime_error("Offline shard tensors have mismatched leading dimensions.");
    }

    torch::Tensor indices = torch::arange(obs.size(0), torch::TensorOptions().dtype(torch::kLong));
    if (shuffle) {
      (void)shard_seed;
      indices = torch::randperm(obs.size(0), torch::TensorOptions().dtype(torch::kLong));
    }

    for (int64_t offset = 0; offset < obs.size(0); offset += batch_size) {
      const int64_t length = std::min<int64_t>(batch_size, obs.size(0) - offset);
      const torch::Tensor batch_indices = indices.narrow(0, offset, length);
      OfflineTensorBatch batch;
      batch.obs = obs.index_select(0, batch_indices);
      if (actions.defined()) {
        batch.actions = actions.index_select(0, batch_indices);
      }
      if (action_probs.defined()) {
        batch.action_probs = action_probs.index_select(0, batch_indices);
      }
      if (next_goal.defined()) {
        batch.next_goal = next_goal.index_select(0, batch_indices);
      }
      batch.weights = weights.index_select(0, batch_indices);
      batch.episode_starts = episode_starts.index_select(0, batch_indices);
      fn(batch);
    }
  }
}

void OfflineTensorDataset::for_each_trajectory(
    bool shuffle,
    std::uint64_t seed,
    const std::function<void(const OfflineTensorBatch&)>& fn) const {
  const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path_).parent_path();
  std::mt19937_64 rng(seed);

  for (const auto& shard : manifest_.shards) {
    torch::Tensor obs = load_tensor_checked((manifest_dir / shard.obs_path).string());
    if (obs.dim() != 2 || obs.size(1) != manifest_.observation_dim) {
      throw std::runtime_error("Offline shard obs tensor has unexpected shape.");
    }

    torch::Tensor actions;
    if (!shard.actions_path.empty()) {
      actions = load_tensor_checked((manifest_dir / shard.actions_path).string()).to(torch::kLong).view({-1});
    }

    torch::Tensor action_probs;
    if (!shard.action_probs_path.empty()) {
      action_probs =
          load_tensor_checked((manifest_dir / shard.action_probs_path).string()).to(torch::kFloat32).contiguous();
    }

    torch::Tensor next_goal;
    if (!shard.next_goal_path.empty()) {
      next_goal = load_tensor_checked((manifest_dir / shard.next_goal_path).string()).to(torch::kLong).view({-1});
    }

    torch::Tensor weights;
    if (!shard.weights_path.empty()) {
      weights = load_tensor_checked((manifest_dir / shard.weights_path).string()).to(torch::kFloat32).view({-1});
    } else {
      weights = torch::ones({obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
    }

    torch::Tensor episode_starts;
    if (!shard.episode_starts_path.empty()) {
      episode_starts =
          load_tensor_checked((manifest_dir / shard.episode_starts_path).string()).to(torch::kFloat32).view({-1});
    } else {
      episode_starts = torch::zeros({obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
      if (obs.size(0) > 0) {
        episode_starts[0] = 1.0F;
      }
    }

    if (obs.size(0) != weights.size(0) ||
        (actions.defined() && obs.size(0) != actions.size(0)) ||
        (action_probs.defined() && (action_probs.dim() != 2 || obs.size(0) != action_probs.size(0))) ||
        (next_goal.defined() && obs.size(0) != next_goal.size(0)) ||
        obs.size(0) != episode_starts.size(0)) {
      throw std::runtime_error("Offline shard tensors have mismatched leading dimensions.");
    }

    const torch::Tensor episode_cpu = episode_starts.to(torch::kCPU).contiguous();
    const float* starts_ptr = episode_cpu.data_ptr<float>();
    std::vector<std::int64_t> trajectory_starts;
    trajectory_starts.reserve(static_cast<std::size_t>(obs.size(0) / 64 + 1));
    if (obs.size(0) > 0 && starts_ptr[0] <= 0.5F) {
      trajectory_starts.push_back(0);
    }
    for (std::int64_t i = 0; i < obs.size(0); ++i) {
      if (starts_ptr[i] > 0.5F) {
        trajectory_starts.push_back(i);
      }
    }
    if (trajectory_starts.empty()) {
      trajectory_starts.push_back(0);
    }

    std::vector<std::int64_t> ordering(trajectory_starts.size());
    std::iota(ordering.begin(), ordering.end(), 0);
    if (shuffle) {
      std::shuffle(ordering.begin(), ordering.end(), rng);
    }

    for (const std::int64_t order_idx : ordering) {
      const std::int64_t start = trajectory_starts[static_cast<std::size_t>(order_idx)];
      const std::int64_t end =
          (static_cast<std::size_t>(order_idx + 1) < trajectory_starts.size())
              ? trajectory_starts[static_cast<std::size_t>(order_idx + 1)]
              : obs.size(0);
      const std::int64_t length = end - start;
      if (length <= 0) {
        continue;
      }

      OfflineTensorBatch batch;
      batch.obs = obs.narrow(0, start, length);
      if (actions.defined()) {
        batch.actions = actions.narrow(0, start, length);
      }
      if (action_probs.defined()) {
        batch.action_probs = action_probs.narrow(0, start, length);
      }
      if (next_goal.defined()) {
        batch.next_goal = next_goal.narrow(0, start, length);
      }
      batch.weights = weights.narrow(0, start, length);
      batch.episode_starts = episode_starts.narrow(0, start, length);
      fn(batch);
    }
  }
}

}  // namespace pulsar

#endif
