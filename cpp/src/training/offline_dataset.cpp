#include "pulsar/training/offline_dataset.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
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

struct LoadedOfflineShard {
  torch::Tensor obs{};
  torch::Tensor actions{};
  torch::Tensor action_probs{};
  torch::Tensor outcome{};
  torch::Tensor outcome_known{};
  torch::Tensor weights{};
  torch::Tensor episode_starts{};
  torch::Tensor terminated{};
  torch::Tensor truncated{};
};

struct OfflineTensorRange {
  std::int64_t start = 0;
  std::int64_t length = 0;
};

LoadedOfflineShard load_shard_tensors(
    const std::filesystem::path& manifest_dir,
    const OfflineTensorShardEntry& shard,
    int observation_dim) {
  LoadedOfflineShard loaded;
  loaded.obs = load_tensor_checked((manifest_dir / shard.obs_path).string()).to(torch::kFloat32).contiguous();
  if (loaded.obs.dim() != 2 || loaded.obs.size(1) != observation_dim) {
    throw std::runtime_error("Offline shard obs tensor has unexpected shape.");
  }

  if (!shard.actions_path.empty()) {
    loaded.actions = load_tensor_checked((manifest_dir / shard.actions_path).string()).to(torch::kLong).view({-1});
  }
  if (!shard.action_probs_path.empty()) {
    loaded.action_probs =
        load_tensor_checked((manifest_dir / shard.action_probs_path).string()).to(torch::kFloat32).contiguous();
  }
  if (!shard.outcome_path.empty()) {
    loaded.outcome = load_tensor_checked((manifest_dir / shard.outcome_path).string()).to(torch::kLong).view({-1});
  }
  if (!shard.outcome_known_path.empty()) {
    loaded.outcome_known =
        load_tensor_checked((manifest_dir / shard.outcome_known_path).string()).to(torch::kFloat32).view({-1});
  } else {
    loaded.outcome_known = loaded.outcome.defined()
        ? torch::ones({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32))
        : torch::zeros({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
  }
  if (!shard.weights_path.empty()) {
    loaded.weights = load_tensor_checked((manifest_dir / shard.weights_path).string()).to(torch::kFloat32).view({-1});
  } else {
    loaded.weights = torch::ones({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
  }
  if (!shard.episode_starts_path.empty()) {
    loaded.episode_starts =
        load_tensor_checked((manifest_dir / shard.episode_starts_path).string()).to(torch::kFloat32).view({-1});
  } else {
    loaded.episode_starts = torch::zeros({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
    if (loaded.obs.size(0) > 0) {
      loaded.episode_starts[0] = 1.0F;
    }
  }
  loaded.terminated = !shard.terminated_path.empty()
      ? load_tensor_checked((manifest_dir / shard.terminated_path).string()).to(torch::kFloat32).view({-1})
      : torch::zeros({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
  loaded.truncated = !shard.truncated_path.empty()
      ? load_tensor_checked((manifest_dir / shard.truncated_path).string()).to(torch::kFloat32).view({-1})
      : torch::zeros({loaded.obs.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));

  if (loaded.obs.size(0) != loaded.weights.size(0) ||
      loaded.obs.size(0) != loaded.outcome_known.size(0) ||
      (loaded.actions.defined() && loaded.obs.size(0) != loaded.actions.size(0)) ||
      (loaded.action_probs.defined() &&
       (loaded.action_probs.dim() != 2 || loaded.obs.size(0) != loaded.action_probs.size(0))) ||
      (loaded.outcome.defined() && loaded.obs.size(0) != loaded.outcome.size(0)) ||
      loaded.obs.size(0) != loaded.episode_starts.size(0) ||
      loaded.obs.size(0) != loaded.terminated.size(0) ||
      loaded.obs.size(0) != loaded.truncated.size(0)) {
    throw std::runtime_error("Offline shard tensors have mismatched leading dimensions.");
  }
  return loaded;
}

std::vector<OfflineTensorRange> build_trajectory_ranges(const torch::Tensor& episode_starts) {
  const torch::Tensor episode_cpu = episode_starts.to(torch::kCPU).contiguous();
  const float* starts_ptr = episode_cpu.data_ptr<float>();
  std::vector<std::int64_t> trajectory_starts;
  if (episode_starts.size(0) > 0 && starts_ptr[0] <= 0.5F) {
    trajectory_starts.push_back(0);
  }
  for (std::int64_t i = 0; i < episode_starts.size(0); ++i) {
    if (starts_ptr[i] > 0.5F) {
      trajectory_starts.push_back(i);
    }
  }
  if (trajectory_starts.empty()) {
    trajectory_starts.push_back(0);
  }
  std::vector<OfflineTensorRange> ranges;
  for (std::size_t i = 0; i < trajectory_starts.size(); ++i) {
    const std::int64_t start = trajectory_starts[i];
    const std::int64_t end = (i + 1 < trajectory_starts.size()) ? trajectory_starts[i + 1] : episode_starts.size(0);
    if (end > start) {
      ranges.push_back({start, end - start});
    }
  }
  return ranges;
}

std::vector<std::int64_t> build_trajectory_order(
    const std::vector<OfflineTensorRange>& ranges,
    bool shuffle,
    std::mt19937_64& rng) {
  std::vector<std::int64_t> ordering(ranges.size());
  std::iota(ordering.begin(), ordering.end(), 0);
  auto compare_by_length = [&](const std::int64_t lhs, const std::int64_t rhs) {
    return ranges[static_cast<std::size_t>(lhs)].length > ranges[static_cast<std::size_t>(rhs)].length;
  };
  if (shuffle) {
    std::shuffle(ordering.begin(), ordering.end(), rng);
    constexpr std::size_t kBucketSize = 32;
    for (std::size_t begin = 0; begin < ordering.size(); begin += kBucketSize) {
      const std::size_t end = std::min<std::size_t>(ordering.size(), begin + kBucketSize);
      std::stable_sort(ordering.begin() + static_cast<std::ptrdiff_t>(begin),
                       ordering.begin() + static_cast<std::ptrdiff_t>(end),
                       compare_by_length);
    }
    return ordering;
  }
  std::stable_sort(ordering.begin(), ordering.end(), compare_by_length);
  return ordering;
}

OfflineTensorPackedBatch pack_trajectory_batch(
    const LoadedOfflineShard& loaded,
    const std::vector<OfflineTensorRange>& ranges,
    const std::vector<std::int64_t>& packed_indices,
    int action_dim) {
  OfflineTensorPackedBatch batch;
  batch.lengths.reserve(packed_indices.size());
  std::int64_t max_length = 0;
  for (const std::int64_t range_index : packed_indices) {
    const auto length = ranges[static_cast<std::size_t>(range_index)].length;
    batch.lengths.push_back(length);
    max_length = std::max(max_length, length);
  }
  const auto packed_count = static_cast<std::int64_t>(packed_indices.size());
  batch.obs = torch::zeros({max_length, packed_count, loaded.obs.size(1)}, loaded.obs.options());
  batch.weights = torch::zeros({max_length, packed_count}, loaded.weights.options());
  batch.outcome_known = torch::zeros({max_length, packed_count}, loaded.outcome_known.options());
  batch.episode_starts = torch::ones({max_length, packed_count}, loaded.episode_starts.options());
  batch.terminated = torch::zeros({max_length, packed_count}, loaded.terminated.options());
  batch.truncated = torch::zeros({max_length, packed_count}, loaded.truncated.options());
  batch.valid_mask = torch::zeros({max_length, packed_count}, torch::TensorOptions().dtype(torch::kBool));
  if (loaded.actions.defined()) {
    batch.actions = torch::zeros({max_length, packed_count}, loaded.actions.options());
  }
  if (loaded.action_probs.defined()) {
    batch.action_probs = torch::zeros({max_length, packed_count, action_dim}, loaded.action_probs.options());
  }
  if (loaded.outcome.defined()) {
    batch.outcome = torch::zeros({max_length, packed_count}, loaded.outcome.options());
  }

  for (std::int64_t packed_column = 0; packed_column < packed_count; ++packed_column) {
    const auto range = ranges[static_cast<std::size_t>(packed_indices[static_cast<std::size_t>(packed_column)])];
    batch.obs.select(1, packed_column).narrow(0, 0, range.length).copy_(loaded.obs.narrow(0, range.start, range.length));
    batch.weights.select(1, packed_column).narrow(0, 0, range.length).copy_(loaded.weights.narrow(0, range.start, range.length));
    batch.outcome_known.select(1, packed_column).narrow(0, 0, range.length).copy_(
        loaded.outcome_known.narrow(0, range.start, range.length));
    batch.episode_starts.select(1, packed_column).narrow(0, 0, range.length).copy_(
        loaded.episode_starts.narrow(0, range.start, range.length));
    batch.terminated.select(1, packed_column).narrow(0, 0, range.length).copy_(
        loaded.terminated.narrow(0, range.start, range.length));
    batch.truncated.select(1, packed_column).narrow(0, 0, range.length).copy_(
        loaded.truncated.narrow(0, range.start, range.length));
    batch.valid_mask.select(1, packed_column).narrow(0, 0, range.length).fill_(true);
    if (loaded.actions.defined()) {
      batch.actions.select(1, packed_column).narrow(0, 0, range.length).copy_(
          loaded.actions.narrow(0, range.start, range.length));
    }
    if (loaded.action_probs.defined()) {
      batch.action_probs.select(1, packed_column).narrow(0, 0, range.length).copy_(
          loaded.action_probs.narrow(0, range.start, range.length));
    }
    if (loaded.outcome.defined()) {
      batch.outcome.select(1, packed_column).narrow(0, 0, range.length).copy_(
          loaded.outcome.narrow(0, range.start, range.length));
    }
  }
  return batch;
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
  manifest.schema_version = j.value("schema_version", 4);
  manifest.observation_dim = j.at("observation_dim").get<int>();
  manifest.action_dim = j.at("action_dim").get<int>();
  manifest.outcome_classes = j.value("outcome_classes", 3);
  for (const auto& shard_json : j.at("shards")) {
    OfflineTensorShardEntry shard;
    shard.obs_path = shard_json.at("obs_path").get<std::string>();
    shard.actions_path = shard_json.value("actions_path", std::string{});
    shard.action_probs_path = shard_json.value("action_probs_path", std::string{});
    shard.outcome_path = shard_json.value("outcome_path", std::string{});
    shard.outcome_known_path = shard_json.value("outcome_known_path", std::string{});
    shard.weights_path = shard_json.value("weights_path", std::string{});
    shard.episode_starts_path = shard_json.value("episode_starts_path", std::string{});
    shard.terminated_path = shard_json.value("terminated_path", std::string{});
    shard.truncated_path = shard_json.value("truncated_path", std::string{});
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

int OfflineTensorDataset::outcome_classes() const {
  return manifest_.outcome_classes;
}

bool OfflineTensorDataset::has_episode_starts() const {
  for (const auto& shard : manifest_.shards) {
    if (shard.episode_starts_path.empty()) {
      return false;
    }
  }
  return !manifest_.shards.empty();
}

bool OfflineTensorDataset::has_trajectory_end_flags() const {
  for (const auto& shard : manifest_.shards) {
    if (shard.terminated_path.empty() || shard.truncated_path.empty()) {
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
    std::uint64_t,
    const std::function<void(const OfflineTensorBatch&)>& fn) const {
  if (batch_size <= 0) {
    throw std::invalid_argument("OfflineTensorDataset batch size must be positive.");
  }
  const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path_).parent_path();
  for (const auto& shard : manifest_.shards) {
    const LoadedOfflineShard loaded = load_shard_tensors(manifest_dir, shard, manifest_.observation_dim);
    torch::Tensor indices = shuffle
        ? torch::randperm(loaded.obs.size(0), torch::TensorOptions().dtype(torch::kLong))
        : torch::arange(loaded.obs.size(0), torch::TensorOptions().dtype(torch::kLong));
    for (int64_t offset = 0; offset < loaded.obs.size(0); offset += batch_size) {
      const int64_t length = std::min<int64_t>(batch_size, loaded.obs.size(0) - offset);
      const torch::Tensor batch_indices = indices.narrow(0, offset, length);
      OfflineTensorBatch batch;
      batch.obs = loaded.obs.index_select(0, batch_indices);
      if (loaded.actions.defined()) batch.actions = loaded.actions.index_select(0, batch_indices);
      if (loaded.action_probs.defined()) batch.action_probs = loaded.action_probs.index_select(0, batch_indices);
      if (loaded.outcome.defined()) batch.outcome = loaded.outcome.index_select(0, batch_indices);
      batch.outcome_known = loaded.outcome_known.index_select(0, batch_indices);
      batch.weights = loaded.weights.index_select(0, batch_indices);
      batch.episode_starts = loaded.episode_starts.index_select(0, batch_indices);
      batch.terminated = loaded.terminated.index_select(0, batch_indices);
      batch.truncated = loaded.truncated.index_select(0, batch_indices);
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
    const LoadedOfflineShard loaded = load_shard_tensors(manifest_dir, shard, manifest_.observation_dim);
    const std::vector<OfflineTensorRange> ranges = build_trajectory_ranges(loaded.episode_starts);
    const std::vector<std::int64_t> ordering = build_trajectory_order(ranges, shuffle, rng);
    for (const std::int64_t order_idx : ordering) {
      const auto range = ranges[static_cast<std::size_t>(order_idx)];
      OfflineTensorBatch batch;
      batch.obs = loaded.obs.narrow(0, range.start, range.length);
      if (loaded.actions.defined()) batch.actions = loaded.actions.narrow(0, range.start, range.length);
      if (loaded.action_probs.defined()) batch.action_probs = loaded.action_probs.narrow(0, range.start, range.length);
      if (loaded.outcome.defined()) batch.outcome = loaded.outcome.narrow(0, range.start, range.length);
      batch.outcome_known = loaded.outcome_known.narrow(0, range.start, range.length);
      batch.weights = loaded.weights.narrow(0, range.start, range.length);
      batch.episode_starts = loaded.episode_starts.narrow(0, range.start, range.length);
      batch.terminated = loaded.terminated.narrow(0, range.start, range.length);
      batch.truncated = loaded.truncated.narrow(0, range.start, range.length);
      fn(batch);
    }
  }
}

void OfflineTensorDataset::for_each_packed_trajectory_batch(
    int max_tokens,
    bool shuffle,
    std::uint64_t seed,
    const std::function<void(const OfflineTensorPackedBatch&)>& fn) const {
  if (max_tokens <= 0) {
    throw std::invalid_argument("OfflineTensorDataset max_tokens must be positive.");
  }
  const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path_).parent_path();
  std::mt19937_64 rng(seed);
  for (const auto& shard : manifest_.shards) {
    const LoadedOfflineShard loaded = load_shard_tensors(manifest_dir, shard, manifest_.observation_dim);
    const std::vector<OfflineTensorRange> ranges = build_trajectory_ranges(loaded.episode_starts);
    const std::vector<std::int64_t> ordering = build_trajectory_order(ranges, shuffle, rng);
    std::vector<std::int64_t> packed;
    std::int64_t packed_tokens = 0;
    for (const std::int64_t range_index : ordering) {
      const std::int64_t length = ranges[static_cast<std::size_t>(range_index)].length;
      if (!packed.empty() && packed_tokens + length > max_tokens) {
        fn(pack_trajectory_batch(loaded, ranges, packed, manifest_.action_dim));
        packed.clear();
        packed_tokens = 0;
      }
      packed.push_back(range_index);
      packed_tokens += length;
    }
    if (!packed.empty()) {
      fn(pack_trajectory_batch(loaded, ranges, packed, manifest_.action_dim));
    }
  }
}

}  // namespace pulsar

#endif
