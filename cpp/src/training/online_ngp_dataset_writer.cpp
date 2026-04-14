#include "pulsar/training/online_ngp_dataset_writer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <limits>

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pulsar {
namespace {

std::uint64_t hash_episode_key(std::uint64_t seed, std::uint64_t env_idx, std::uint64_t episode_id) {
  std::uint64_t value = 1469598103934665603ULL;
  for (const std::uint64_t part : {seed, env_idx, episode_id}) {
    value ^= part;
    value *= 1099511628211ULL;
  }
  return value;
}

}  // namespace

OnlineNGPDatasetWriter::OnlineNGPDatasetWriter(
    RewardConfig::OnlineDatasetExportConfig config,
    std::filesystem::path output_root,
    int obs_dim,
    int action_dim,
    std::size_t num_envs,
    std::size_t agents_per_env)
    : config_(std::move(config)),
      output_root_(std::move(output_root)),
      obs_dim_(obs_dim),
      action_dim_(action_dim),
      num_envs_(num_envs),
      agents_per_env_(agents_per_env),
      trajectories_(num_envs * agents_per_env),
      env_episode_ids_(num_envs, 0) {
  if (obs_dim_ <= 0) {
    throw std::invalid_argument("OnlineNGPDatasetWriter requires a positive obs_dim.");
  }
  if (action_dim_ <= 0) {
    throw std::invalid_argument("OnlineNGPDatasetWriter requires a positive action_dim.");
  }
  if (num_envs_ == 0 || agents_per_env_ == 0) {
    throw std::invalid_argument("OnlineNGPDatasetWriter requires non-zero environment counts.");
  }
  if (config_.shard_size <= 0) {
    throw std::invalid_argument("OnlineNGPDatasetWriter shard_size must be positive.");
  }
  if (!(config_.train_fraction >= 0.0F && config_.train_fraction <= 1.0F)) {
    throw std::invalid_argument("OnlineNGPDatasetWriter train_fraction must be between 0 and 1.");
  }

  std::filesystem::create_directories(output_root_);
}

void OnlineNGPDatasetWriter::record_step(
    const torch::Tensor& raw_obs_cpu,
    const torch::Tensor& dones_cpu,
    const torch::Tensor& terminal_next_goal_labels_cpu) {
  if (raw_obs_cpu.device().type() != torch::kCPU || dones_cpu.device().type() != torch::kCPU ||
      terminal_next_goal_labels_cpu.device().type() != torch::kCPU) {
    throw std::invalid_argument("OnlineNGPDatasetWriter expects CPU tensors.");
  }
  if (raw_obs_cpu.dim() != 2 || raw_obs_cpu.size(1) != obs_dim_) {
    throw std::invalid_argument("OnlineNGPDatasetWriter raw_obs tensor has the wrong shape.");
  }
  const std::size_t total_agents = num_envs_ * agents_per_env_;
  if (static_cast<std::size_t>(raw_obs_cpu.size(0)) != total_agents ||
      static_cast<std::size_t>(dones_cpu.numel()) != total_agents ||
      static_cast<std::size_t>(terminal_next_goal_labels_cpu.numel()) != total_agents) {
    throw std::invalid_argument("OnlineNGPDatasetWriter tensors disagree on agent count.");
  }

  const auto obs = raw_obs_cpu.contiguous();
  const auto dones = dones_cpu.contiguous();
  const auto labels = terminal_next_goal_labels_cpu.contiguous();
  const float* obs_ptr = obs.data_ptr<float>();
  const float* dones_ptr = dones.data_ptr<float>();
  const std::int64_t* labels_ptr = labels.data_ptr<std::int64_t>();

  for (std::size_t env_idx = 0; env_idx < num_envs_; ++env_idx) {
    const std::string split = choose_split(env_idx);
    SplitBuffers& buffers = split == "train" ? train_ : val_;
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

    for (std::size_t local_idx = 0; local_idx < agents_per_env_; ++local_idx) {
      const std::size_t agent_idx = env_base + local_idx;
      AgentTrajectory& trajectory = trajectories_[agent_idx];
      if (trajectory.steps == 0) {
        continue;
      }

      buffers.obs.insert(buffers.obs.end(), trajectory.obs.begin(), trajectory.obs.end());
      buffers.next_goal.insert(buffers.next_goal.end(), trajectory.steps, labels_ptr[agent_idx]);
      buffers.weights.insert(buffers.weights.end(), trajectory.steps, 1.0F);
      buffers.episode_starts.push_back(1.0F);
      if (trajectory.steps > 1) {
        buffers.episode_starts.insert(buffers.episode_starts.end(), trajectory.steps - 1, 0.0F);
      }
      samples_written_ += static_cast<std::int64_t>(trajectory.steps);
      trajectories_written_ += 1;
      trajectory.obs.clear();
      trajectory.steps = 0;
    }

    env_episode_ids_[env_idx] += 1;
    flush_ready_split(split);
  }
}

void OnlineNGPDatasetWriter::finish() {
  flush_split(train_, "train");
  flush_split(val_, "val");
  write_manifest(train_, output_root_ / "train_manifest.json");
  write_manifest(val_.shards.empty() ? train_ : val_, output_root_ / "val_manifest.json");
}

std::int64_t OnlineNGPDatasetWriter::samples_written() const {
  return samples_written_;
}

std::int64_t OnlineNGPDatasetWriter::trajectories_written() const {
  return trajectories_written_;
}

std::string OnlineNGPDatasetWriter::choose_split(std::size_t env_idx) const {
  const std::uint64_t draw =
      hash_episode_key(config_.seed, static_cast<std::uint64_t>(env_idx), env_episode_ids_[env_idx]);
  const double fraction = static_cast<double>(draw) / static_cast<double>(std::numeric_limits<std::uint64_t>::max());
  return fraction < static_cast<double>(config_.train_fraction) ? "train" : "val";
}

void OnlineNGPDatasetWriter::flush_ready_split(const std::string& split) {
  SplitBuffers& buffers = split == "train" ? train_ : val_;
  const std::int64_t sample_count =
      static_cast<std::int64_t>(buffers.next_goal.size());
  if (sample_count >= config_.shard_size) {
    flush_split(buffers, split);
  }
}

void OnlineNGPDatasetWriter::flush_split(SplitBuffers& buffers, const std::string& split) {
  if (buffers.next_goal.empty()) {
    return;
  }

  const std::filesystem::path split_dir = output_root_ / split;
  std::filesystem::create_directories(split_dir);
  const std::string shard_name =
      "shard_" + std::to_string(static_cast<long long>(buffers.shard_index++));

  const auto obs = torch::from_blob(
                       buffers.obs.data(),
                       {static_cast<long>(buffers.next_goal.size()), static_cast<long>(obs_dim_)},
                       torch::TensorOptions().dtype(torch::kFloat32))
                       .clone();
  const auto next_goal = torch::tensor(buffers.next_goal, torch::TensorOptions().dtype(torch::kLong));
  const auto weights = torch::tensor(buffers.weights, torch::TensorOptions().dtype(torch::kFloat32));
  const auto episode_starts =
      torch::tensor(buffers.episode_starts, torch::TensorOptions().dtype(torch::kFloat32));

  const auto obs_path = (split_dir / (shard_name + "_obs.pt")).string();
  const auto next_goal_path = (split_dir / (shard_name + "_next_goal.pt")).string();
  const auto weights_path = (split_dir / (shard_name + "_weights.pt")).string();
  const auto episode_starts_path = (split_dir / (shard_name + "_episode_starts.pt")).string();

  torch::save(obs, obs_path);
  torch::save(next_goal, next_goal_path);
  torch::save(weights, weights_path);
  torch::save(episode_starts, episode_starts_path);

  nlohmann::json entry = {
      {"obs_path", obs_path},
      {"next_goal_path", next_goal_path},
      {"weights_path", weights_path},
      {"episode_starts_path", episode_starts_path},
      {"samples", static_cast<std::int64_t>(buffers.next_goal.size())},
  };
  buffers.shards.push_back(entry.dump());

  buffers.obs.clear();
  buffers.next_goal.clear();
  buffers.weights.clear();
  buffers.episode_starts.clear();

  write_manifest(train_, output_root_ / "train_manifest.json");
  write_manifest(val_.shards.empty() ? train_ : val_, output_root_ / "val_manifest.json");
}

void OnlineNGPDatasetWriter::write_manifest(const SplitBuffers& buffers, const std::filesystem::path& path) const {
  nlohmann::json manifest = {
      {"schema_version", 3},
      {"observation_dim", obs_dim_},
      {"action_dim", action_dim_},
      {"next_goal_classes", 3},
      {"shards", nlohmann::json::array()},
  };
  for (const auto& shard : buffers.shards) {
    manifest["shards"].push_back(nlohmann::json::parse(shard));
  }

  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to write online NGP manifest: " + path.string());
  }
  output << manifest.dump(2) << '\n';
}

}  // namespace pulsar

#endif
