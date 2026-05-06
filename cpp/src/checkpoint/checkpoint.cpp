#include "pulsar/checkpoint/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <system_error>

#include <nlohmann/json.hpp>

namespace pulsar {
namespace {

constexpr std::array<const char*, 5> kCheckpointFiles = {
    "model.pt",
    "metadata.json",
    "config.json",
    "ratings.json",
    "actor_optimizer.pt",
};

std::filesystem::path unique_sibling_path(const std::filesystem::path& target, const char* infix) {
  static std::atomic<std::uint64_t> counter{0};
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return target.parent_path() /
      (target.filename().string() + infix + "." + std::to_string(stamp) + "." +
       std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
}

void validate_critic_heads(const CheckpointMetadata& metadata) {
  if (metadata.critic_heads.empty()) {
    throw std::runtime_error("Checkpoint critic_heads metadata must not be empty.");
  }

  const bool has_extrinsic = std::find(
      metadata.critic_heads.begin(),
      metadata.critic_heads.end(),
      std::string{"extrinsic"}) != metadata.critic_heads.end();
  if (!has_extrinsic) {
    throw std::runtime_error("Checkpoint critic_heads metadata must include the extrinsic head.");
  }
}

}  // namespace

void save_checkpoint_metadata(const CheckpointMetadata& metadata, const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to write checkpoint metadata: " + path);
  }

  nlohmann::json j = nlohmann::json::parse(stable_json(metadata));
  output << std::setw(2) << j << '\n';
}

CheckpointMetadata load_checkpoint_metadata(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to read checkpoint metadata: " + path);
  }

  nlohmann::json j;
  input >> j;
  return j.get<CheckpointMetadata>();
}

void validate_checkpoint_metadata(const CheckpointMetadata& metadata, const ExperimentConfig& config) {
  validate_critic_heads(metadata);
  if (metadata.schema_version != config.schema_version) {
    throw std::runtime_error("Checkpoint schema_version does not match config.");
  }
  if (metadata.obs_schema_version != config.obs_schema_version) {
    throw std::runtime_error("Checkpoint obs_schema_version does not match config.");
  }
  if (metadata.config_hash != config_hash(config)) {
    throw std::runtime_error("Checkpoint config hash does not match the active config.");
  }
  if (metadata.action_table_hash != action_table_hash(config.action_table)) {
    throw std::runtime_error("Checkpoint action table hash does not match the active config.");
  }
}

void validate_inference_checkpoint_metadata(const CheckpointMetadata& metadata, const ExperimentConfig& config) {
  validate_critic_heads(metadata);
  if (metadata.schema_version != config.schema_version) {
    throw std::runtime_error("Checkpoint schema_version does not match config.");
  }
  if (metadata.obs_schema_version != config.obs_schema_version) {
    throw std::runtime_error("Checkpoint obs_schema_version does not match config.");
  }
  if (metadata.action_table_hash != action_table_hash(config.action_table)) {
    throw std::runtime_error("Checkpoint action table hash does not match the active config.");
  }
}

std::filesystem::path make_checkpoint_staging_directory(const std::filesystem::path& target) {
  return unique_sibling_path(target, ".tmp");
}

void remove_checkpoint_directory(const std::filesystem::path& directory) noexcept {
  std::error_code ec;
  if (directory.empty() || !std::filesystem::exists(directory, ec)) {
    return;
  }
  if (!std::filesystem::is_directory(directory, ec)) {
    std::filesystem::remove(directory, ec);
    return;
  }
  for (const char* file_name : kCheckpointFiles) {
    std::filesystem::remove(directory / file_name, ec);
    ec.clear();
  }
  std::filesystem::remove(directory, ec);
}

void commit_checkpoint_directory(
    const std::filesystem::path& staging_directory,
    const std::filesystem::path& target_directory) {
  std::error_code ec;
  const std::filesystem::path parent = target_directory.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      throw std::runtime_error("Failed to create checkpoint parent directory: " + parent.string());
    }
  }

  std::filesystem::path stale_directory;
  if (std::filesystem::exists(target_directory, ec)) {
    ec.clear();
    stale_directory = unique_sibling_path(target_directory, ".stale");
    std::filesystem::rename(target_directory, stale_directory, ec);
    if (ec) {
      ec.clear();
      remove_checkpoint_directory(target_directory);
      stale_directory.clear();
    }
  }

  std::filesystem::rename(staging_directory, target_directory, ec);
  if (ec) {
    ec.clear();
    remove_checkpoint_directory(target_directory);
    std::filesystem::rename(staging_directory, target_directory, ec);
  }
  if (ec) {
    throw std::runtime_error("Failed to commit checkpoint directory: " + target_directory.string());
  }

  if (!stale_directory.empty()) {
    remove_checkpoint_directory(stale_directory);
  }
}

}  // namespace pulsar
