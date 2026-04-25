#include "pulsar/training/online_ngp_replay_buffer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <torch/serialize.h>

#include "pulsar/training/offline_dataset.hpp"

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

torch::Tensor pack_trajectory_obs(
    const std::vector<NGPTrajectory>& trajectories,
    int obs_dim,
    std::vector<std::int64_t>* lengths,
    std::vector<std::int64_t>* labels) {
  std::int64_t total_steps = 0;
  lengths->clear();
  labels->clear();
  for (const auto& trajectory : trajectories) {
    const std::int64_t steps = trajectory.obs_cpu.defined() ? trajectory.obs_cpu.size(0) : 0;
    lengths->push_back(steps);
    labels->push_back(trajectory.label);
    total_steps += steps;
  }
  if (total_steps <= 0) {
    return torch::zeros({0, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

  torch::Tensor packed = torch::zeros({total_steps, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  std::int64_t offset = 0;
  for (const auto& trajectory : trajectories) {
    const std::int64_t steps = trajectory.obs_cpu.size(0);
    packed.narrow(0, offset, steps).copy_(trajectory.obs_cpu);
    offset += steps;
  }
  return packed;
}

std::vector<NGPTrajectory> unpack_trajectory_obs(
    const torch::Tensor& packed_obs,
    const torch::Tensor& lengths,
    const torch::Tensor& labels) {
  std::vector<NGPTrajectory> trajectories;
  const torch::Tensor lengths_cpu = lengths.to(torch::kCPU).contiguous();
  const torch::Tensor labels_cpu = labels.to(torch::kCPU).contiguous();
  const auto* lengths_ptr = lengths_cpu.data_ptr<std::int64_t>();
  const auto* labels_ptr = labels_cpu.data_ptr<std::int64_t>();
  std::int64_t offset = 0;
  for (std::int64_t i = 0; i < lengths_cpu.numel(); ++i) {
    const std::int64_t steps = lengths_ptr[i];
    NGPTrajectory trajectory;
    trajectory.label = labels_ptr[i];
    if (steps > 0) {
      trajectory.obs_cpu = packed_obs.narrow(0, offset, steps).clone().to(torch::kCPU);
      offset += steps;
    } else {
      trajectory.obs_cpu = torch::zeros({0, packed_obs.size(1)}, torch::TensorOptions().dtype(torch::kFloat32));
    }
    trajectories.push_back(std::move(trajectory));
  }
  return trajectories;
}

torch::Tensor pack_partial_obs(
    const std::vector<OnlineNGPReplayBuffer::AgentTrajectory>& trajectories,
    int obs_dim,
    std::vector<std::int64_t>* lengths) {
  std::int64_t total_steps = 0;
  lengths->clear();
  for (const auto& trajectory : trajectories) {
    const std::int64_t steps = static_cast<std::int64_t>(trajectory.steps);
    lengths->push_back(steps);
    total_steps += steps;
  }
  if (total_steps <= 0) {
    return torch::zeros({0, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

  torch::Tensor packed = torch::zeros({total_steps, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  std::int64_t offset = 0;
  for (const auto& trajectory : trajectories) {
    const std::int64_t steps = static_cast<std::int64_t>(trajectory.steps);
    if (steps <= 0) {
      continue;
    }
    const torch::Tensor current =
        torch::from_blob(
            const_cast<float*>(trajectory.obs.data()),
            {steps, obs_dim},
            torch::TensorOptions().dtype(torch::kFloat32))
            .clone();
    packed.narrow(0, offset, steps).copy_(current);
    offset += steps;
  }
  return packed;
}

void unpack_partial_obs(
    const torch::Tensor& packed_obs,
    const torch::Tensor& lengths,
    int obs_dim,
    std::vector<OnlineNGPReplayBuffer::AgentTrajectory>* trajectories) {
  trajectories->assign(static_cast<std::size_t>(lengths.numel()), OnlineNGPReplayBuffer::AgentTrajectory{});
  const torch::Tensor lengths_cpu = lengths.to(torch::kCPU).contiguous();
  const auto* lengths_ptr = lengths_cpu.data_ptr<std::int64_t>();
  std::int64_t offset = 0;
  for (std::int64_t i = 0; i < lengths_cpu.numel(); ++i) {
    const std::int64_t steps = lengths_ptr[i];
    auto& trajectory = (*trajectories)[static_cast<std::size_t>(i)];
    trajectory.steps = static_cast<std::size_t>(steps);
    if (steps <= 0) {
      continue;
    }
    const torch::Tensor current = packed_obs.narrow(0, offset, steps).contiguous().to(torch::kCPU);
    const auto* data = current.data_ptr<float>();
    trajectory.obs.assign(data, data + (steps * obs_dim));
    offset += steps;
  }
}

}  // namespace

std::int64_t ngp_trajectory_sample_count(const std::vector<NGPTrajectory>& trajectories) {
  std::int64_t samples = 0;
  for (const auto& trajectory : trajectories) {
    samples += trajectory.obs_cpu.defined() ? trajectory.obs_cpu.size(0) : 0;
  }
  return samples;
}

std::vector<NGPTrajectory> load_ngp_trajectories_from_manifest(const std::string& manifest_path) {
  const OfflineTensorManifest manifest = load_offline_tensor_manifest(manifest_path);
  const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path).parent_path();
  std::vector<NGPTrajectory> trajectories;

  for (const auto& shard : manifest.shards) {
    torch::Tensor obs = load_tensor_checked((manifest_dir / shard.obs_path).string()).to(torch::kFloat32).contiguous();
    torch::Tensor next_goal =
        load_tensor_checked((manifest_dir / shard.next_goal_path).string()).to(torch::kLong).view({-1});
    torch::Tensor episode_starts =
        load_tensor_checked((manifest_dir / shard.episode_starts_path).string()).to(torch::kFloat32).view({-1});

    const torch::Tensor starts_cpu = episode_starts.to(torch::kCPU).contiguous();
    const auto* starts_ptr = starts_cpu.data_ptr<float>();
    std::vector<std::int64_t> trajectory_starts;
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

    for (std::size_t idx = 0; idx < trajectory_starts.size(); ++idx) {
      const std::int64_t start = trajectory_starts[idx];
      const std::int64_t end =
          idx + 1 < trajectory_starts.size() ? trajectory_starts[idx + 1] : obs.size(0);
      const std::int64_t length = end - start;
      if (length <= 0) {
        continue;
      }
      NGPTrajectory trajectory;
      trajectory.obs_cpu = obs.narrow(0, start, length).clone().to(torch::kCPU);
      trajectory.label = next_goal[start].item<std::int64_t>();
      trajectories.push_back(std::move(trajectory));
    }
  }

  return trajectories;
}

std::vector<NGPTrajectory> select_ngp_trajectory_subset(
    const std::vector<NGPTrajectory>& trajectories,
    std::int64_t target_samples,
    std::uint64_t seed) {
  if (target_samples <= 0 || trajectories.empty()) {
    return {};
  }
  if (target_samples >= ngp_trajectory_sample_count(trajectories)) {
    return trajectories;
  }

  std::vector<std::size_t> indices(trajectories.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937_64 rng(seed);
  std::shuffle(indices.begin(), indices.end(), rng);

  std::vector<NGPTrajectory> selected;
  std::int64_t samples = 0;
  for (const std::size_t index : indices) {
    selected.push_back(trajectories[index]);
    samples += trajectories[index].obs_cpu.size(0);
    if (samples >= target_samples) {
      break;
    }
  }
  return selected;
}

void AnchorManifest::build(const std::string& manifest_path, int obs_dim) {
  manifest_path_ = manifest_path;
  obs_dim_ = obs_dim;
  manifest_ = load_offline_tensor_manifest(manifest_path_);

  index_.clear();
  total_samples_ = 0;
  for (const auto& shard : manifest_.shards) {
    total_samples_ += shard.samples;
  }
}

bool AnchorManifest::empty() const {
  return manifest_.shards.empty() || total_samples_ <= 0;
}

std::int64_t AnchorManifest::total_samples() const {
  return total_samples_;
}

std::int64_t AnchorManifest::num_trajectories() const {
  return static_cast<std::int64_t>(index_.size());
}

std::vector<NGPTrajectory> AnchorManifest::sample(
    std::int64_t target_samples,
    std::uint64_t seed) const {
  if (target_samples <= 0 || manifest_.shards.empty()) {
    return {};
  }
  if (target_samples >= total_samples_) {
    return load_all();
  }

  std::vector<std::size_t> shard_order;
  shard_order.reserve(manifest_.shards.size());
  for (std::size_t shard_idx = 0; shard_idx < manifest_.shards.size(); ++shard_idx) {
    if (manifest_.shards[shard_idx].samples > 0) {
      shard_order.push_back(shard_idx);
    }
  }

  std::mt19937_64 rng(seed);
  std::shuffle(shard_order.begin(), shard_order.end(), rng);

  const std::filesystem::path manifest_dir = std::filesystem::path(manifest_path_).parent_path();
  std::vector<NGPTrajectory> result;
  std::int64_t samples = 0;

  for (const std::size_t shard_idx : shard_order) {
    const auto& shard = manifest_.shards[shard_idx];
    const torch::Tensor episode_starts =
        load_tensor_checked((manifest_dir / shard.episode_starts_path).string())
            .to(torch::kFloat32)
            .view({-1});
    const torch::Tensor next_goal =
        load_tensor_checked((manifest_dir / shard.next_goal_path).string())
            .to(torch::kLong)
            .view({-1});
    const torch::Tensor obs =
        load_tensor_checked((manifest_dir / shard.obs_path).string()).to(torch::kFloat32);

    const torch::Tensor starts_cpu = episode_starts.to(torch::kCPU).contiguous();
    const float* starts_ptr = starts_cpu.data_ptr<float>();
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

    std::vector<std::size_t> local_order(trajectory_starts.size());
    std::iota(local_order.begin(), local_order.end(), 0);
    std::shuffle(local_order.begin(), local_order.end(), rng);

    for (const std::size_t local_idx : local_order) {
      const std::int64_t start = trajectory_starts[local_idx];
      const std::int64_t end =
          local_idx + 1 < trajectory_starts.size() ? trajectory_starts[local_idx + 1] : obs.size(0);
      const std::int64_t length = end - start;
      if (length <= 0) {
        continue;
      }
      NGPTrajectory traj;
      traj.obs_cpu = obs.narrow(0, start, length).clone().to(torch::kCPU);
      traj.label = next_goal[start].item<std::int64_t>();
      samples += length;
      result.push_back(std::move(traj));
      if (samples >= target_samples) {
        return result;
      }
    }
  }

  return result;
}

std::vector<NGPTrajectory> AnchorManifest::load_all() const {
  return load_ngp_trajectories_from_manifest(manifest_path_);
}

OnlineNGPReplayBuffer::OnlineNGPReplayBuffer(
    RewardConfig::OnlineDatasetExportConfig export_config,
    RewardConfig::RefreshConfig refresh_config,
    int obs_dim,
    std::size_t num_envs,
    std::size_t agents_per_env)
    : export_config_(std::move(export_config)),
      refresh_config_(std::move(refresh_config)),
      obs_dim_(obs_dim),
      num_envs_(num_envs),
      agents_per_env_(agents_per_env),
      trajectories_(num_envs * agents_per_env),
      env_episode_ids_(num_envs, 0) {
  if (obs_dim_ <= 0) {
    throw std::invalid_argument("OnlineNGPReplayBuffer requires a positive obs_dim.");
  }
  if (num_envs_ == 0 || agents_per_env_ == 0) {
    throw std::invalid_argument("OnlineNGPReplayBuffer requires non-zero environment counts.");
  }
  current_window_.id = next_window_id_++;
}

void OnlineNGPReplayBuffer::record_step(
    const torch::Tensor& raw_obs_cpu,
    const torch::Tensor& dones_cpu,
    const torch::Tensor& terminated_cpu,
    const torch::Tensor& truncated_cpu,
    const torch::Tensor& terminal_next_goal_labels_cpu) {
  if (raw_obs_cpu.device().type() != torch::kCPU || dones_cpu.device().type() != torch::kCPU ||
      terminated_cpu.device().type() != torch::kCPU ||
      truncated_cpu.device().type() != torch::kCPU ||
      terminal_next_goal_labels_cpu.device().type() != torch::kCPU) {
    throw std::invalid_argument("OnlineNGPReplayBuffer expects CPU tensors.");
  }
  const auto obs = raw_obs_cpu.contiguous();
  const auto dones = dones_cpu.contiguous();
  const auto labels = terminal_next_goal_labels_cpu.contiguous();
  const float* obs_ptr = obs.data_ptr<float>();
  const float* dones_ptr = dones.data_ptr<float>();
  const std::int64_t* labels_ptr = labels.data_ptr<std::int64_t>();

  for (std::size_t env_idx = 0; env_idx < num_envs_; ++env_idx) {
    const std::size_t env_base = env_idx * agents_per_env_;

    for (std::size_t local_idx = 0; local_idx < agents_per_env_; ++local_idx) {
      const std::size_t agent_idx = env_base + local_idx;
      AgentTrajectory& trajectory = trajectories_[agent_idx];
      const float* row = obs_ptr + static_cast<std::ptrdiff_t>(agent_idx * static_cast<std::size_t>(obs_dim_));
      trajectory.obs.insert(trajectory.obs.end(), row, row + obs_dim_);
      trajectory.steps += 1;
    }

    const bool episode_done = dones_ptr[env_base] > 0.5F;
    if (!episode_done) {
      continue;
    }

    const std::string split = choose_split();
    auto* destination = split == "train" ? &current_window_.train : &current_window_.val;
    auto* destination_samples = split == "train" ? &current_window_.train_samples : &current_window_.val_samples;
    for (std::size_t local_idx = 0; local_idx < agents_per_env_; ++local_idx) {
      const std::size_t agent_idx = env_base + local_idx;
      AgentTrajectory& partial = trajectories_[agent_idx];
      if (partial.steps == 0) {
        continue;
      }
      NGPTrajectory completed;
      completed.obs_cpu =
          torch::from_blob(
              partial.obs.data(),
              {static_cast<long>(partial.steps), static_cast<long>(obs_dim_)},
              torch::TensorOptions().dtype(torch::kFloat32))
              .clone();
      completed.label = labels_ptr[agent_idx];
      destination->push_back(std::move(completed));
      *destination_samples += static_cast<std::int64_t>(partial.steps);
      partial.obs.clear();
      partial.steps = 0;
      trajectories_written_ += 1;
    }

    env_episode_ids_[env_idx] += 1;
    completed_episodes_ += 1;
  }
}

void OnlineNGPReplayBuffer::close_window() {
  if (current_window_.train_samples > 0 || current_window_.val_samples > 0) {
    windows_.push_back(std::move(current_window_));
    enforce_retention_limits();
  }
  current_window_ = Window{};
  current_window_.id = next_window_id_++;
}

void OnlineNGPReplayBuffer::clear_completed_windows() {
  windows_.clear();
}

std::vector<NGPTrajectory> OnlineNGPReplayBuffer::train_trajectories() const {
  std::vector<NGPTrajectory> merged;
  for (const auto& window : windows_) {
    merged.insert(merged.end(), window.train.begin(), window.train.end());
  }
  return merged;
}

std::vector<NGPTrajectory> OnlineNGPReplayBuffer::val_trajectories() const {
  std::vector<NGPTrajectory> merged;
  for (const auto& window : windows_) {
    merged.insert(merged.end(), window.val.begin(), window.val.end());
  }
  return merged;
}

std::int64_t OnlineNGPReplayBuffer::train_sample_count() const {
  std::int64_t total = 0;
  for (const auto& window : windows_) {
    total += window.train_samples;
  }
  return total;
}

std::int64_t OnlineNGPReplayBuffer::val_sample_count() const {
  std::int64_t total = 0;
  for (const auto& window : windows_) {
    total += window.val_samples;
  }
  return total;
}

std::int64_t OnlineNGPReplayBuffer::trajectories_written() const {
  return trajectories_written_;
}

std::size_t OnlineNGPReplayBuffer::retained_window_count() const {
  return windows_.size();
}

std::shared_ptr<OnlineNGPReplayBuffer> OnlineNGPReplayBuffer::clone() const {
  return std::make_shared<OnlineNGPReplayBuffer>(*this);
}

std::string OnlineNGPReplayBuffer::choose_split() const {
  const double train_fraction =
      std::clamp(static_cast<double>(refresh_config_.online_train_fraction), 0.0, 1.0);
  const double completed = static_cast<double>(completed_episodes_);
  const bool assign_train =
      std::floor((completed + 1.0) * train_fraction) > std::floor(completed * train_fraction);
  return assign_train ? "train" : "val";
}

void OnlineNGPReplayBuffer::enforce_retention_limits() {
  while (!windows_.empty()) {
    const bool over_windows =
        refresh_config_.max_online_windows > 0 &&
        static_cast<int>(windows_.size()) > refresh_config_.max_online_windows;
    const bool over_samples =
        refresh_config_.max_online_samples > 0 &&
        (train_sample_count() + val_sample_count()) > refresh_config_.max_online_samples;
    if (!over_windows && !over_samples) {
      break;
    }
    windows_.pop_front();
  }
}

void OnlineNGPReplayBuffer::save_window(
    const Window& window,
    const std::filesystem::path& directory,
    const std::string& prefix) const {
  std::vector<std::int64_t> train_lengths;
  std::vector<std::int64_t> train_labels;
  const torch::Tensor train_obs = pack_trajectory_obs(window.train, obs_dim_, &train_lengths, &train_labels);
  const torch::Tensor train_lengths_tensor = torch::tensor(train_lengths, torch::TensorOptions().dtype(torch::kLong));
  const torch::Tensor train_labels_tensor = torch::tensor(train_labels, torch::TensorOptions().dtype(torch::kLong));

  std::vector<std::int64_t> val_lengths;
  std::vector<std::int64_t> val_labels;
  const torch::Tensor val_obs = pack_trajectory_obs(window.val, obs_dim_, &val_lengths, &val_labels);
  const torch::Tensor val_lengths_tensor = torch::tensor(val_lengths, torch::TensorOptions().dtype(torch::kLong));
  const torch::Tensor val_labels_tensor = torch::tensor(val_labels, torch::TensorOptions().dtype(torch::kLong));

  torch::save(train_obs, directory / (prefix + "_train_obs.pt"));
  torch::save(train_lengths_tensor, directory / (prefix + "_train_lengths.pt"));
  torch::save(train_labels_tensor, directory / (prefix + "_train_labels.pt"));
  torch::save(val_obs, directory / (prefix + "_val_obs.pt"));
  torch::save(val_lengths_tensor, directory / (prefix + "_val_lengths.pt"));
  torch::save(val_labels_tensor, directory / (prefix + "_val_labels.pt"));
}

void OnlineNGPReplayBuffer::load_window(
    Window* window,
    const std::filesystem::path& directory,
    const std::string& prefix) const {
  torch::Tensor train_obs;
  torch::Tensor train_lengths;
  torch::Tensor train_labels;
  torch::Tensor val_obs;
  torch::Tensor val_lengths;
  torch::Tensor val_labels;
  torch::load(train_obs, directory / (prefix + "_train_obs.pt"));
  torch::load(train_lengths, directory / (prefix + "_train_lengths.pt"));
  torch::load(train_labels, directory / (prefix + "_train_labels.pt"));
  torch::load(val_obs, directory / (prefix + "_val_obs.pt"));
  torch::load(val_lengths, directory / (prefix + "_val_lengths.pt"));
  torch::load(val_labels, directory / (prefix + "_val_labels.pt"));
  window->train = unpack_trajectory_obs(train_obs, train_lengths, train_labels);
  window->val = unpack_trajectory_obs(val_obs, val_lengths, val_labels);
  window->train_samples = ngp_trajectory_sample_count(window->train);
  window->val_samples = ngp_trajectory_sample_count(window->val);
}

void OnlineNGPReplayBuffer::save(const std::filesystem::path& directory) const {
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);

  json metadata = {
      {"obs_dim", obs_dim_},
      {"num_envs", num_envs_},
      {"agents_per_env", agents_per_env_},
      {"next_window_id", next_window_id_},
      {"trajectories_written", trajectories_written_},
      {"env_episode_ids", env_episode_ids_},
      {"completed_episodes", completed_episodes_},
      {"window_ids", json::array()},
      {"current_window_id", current_window_.id},
  };
  for (const auto& window : windows_) {
    metadata["window_ids"].push_back(window.id);
  }

  save_window(current_window_, directory, "current");
  for (const auto& window : windows_) {
    save_window(window, directory, "window_" + std::to_string(window.id));
  }

  std::vector<std::int64_t> partial_lengths;
  const torch::Tensor partial_obs = pack_partial_obs(trajectories_, obs_dim_, &partial_lengths);
  torch::save(partial_obs, directory / "partial_obs.pt");
  torch::save(
      torch::tensor(partial_lengths, torch::TensorOptions().dtype(torch::kLong)),
      directory / "partial_lengths.pt");

  std::ofstream output(directory / "metadata.json");
  output << metadata.dump(2) << '\n';
}

void OnlineNGPReplayBuffer::load(const std::filesystem::path& directory) {
  std::ifstream input(directory / "metadata.json");
  if (!input) {
    throw std::runtime_error("Failed to load replay buffer metadata: " + (directory / "metadata.json").string());
  }
  json metadata;
  input >> metadata;

  env_episode_ids_ = metadata.at("env_episode_ids").get<std::vector<std::uint64_t>>();
  completed_episodes_ = metadata.value("completed_episodes", static_cast<std::uint64_t>(0));
  next_window_id_ = metadata.at("next_window_id").get<std::int64_t>();
  trajectories_written_ = metadata.value("trajectories_written", static_cast<std::int64_t>(0));

  windows_.clear();
  for (const auto& window_id_json : metadata.at("window_ids")) {
    Window window;
    window.id = window_id_json.get<std::int64_t>();
    load_window(&window, directory, "window_" + std::to_string(window.id));
    windows_.push_back(std::move(window));
  }

  current_window_ = Window{};
  current_window_.id = metadata.at("current_window_id").get<std::int64_t>();
  load_window(&current_window_, directory, "current");

  torch::Tensor partial_obs;
  torch::Tensor partial_lengths;
  torch::load(partial_obs, directory / "partial_obs.pt");
  torch::load(partial_lengths, directory / "partial_lengths.pt");
  unpack_partial_obs(partial_obs, partial_lengths, obs_dim_, &trajectories_);
}

}  // namespace pulsar

#endif
