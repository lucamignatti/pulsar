#include "pulsar/checkpoint/checkpoint.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pulsar {
namespace {

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

}  // namespace pulsar
