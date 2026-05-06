#pragma once

#include <filesystem>
#include <string>

#include "pulsar/config/config.hpp"

namespace pulsar {

void save_checkpoint_metadata(const CheckpointMetadata& metadata, const std::string& path);
CheckpointMetadata load_checkpoint_metadata(const std::string& path);
void validate_checkpoint_metadata(const CheckpointMetadata& metadata, const ExperimentConfig& config);
void validate_inference_checkpoint_metadata(const CheckpointMetadata& metadata, const ExperimentConfig& config);
std::filesystem::path make_checkpoint_staging_directory(const std::filesystem::path& target);
void commit_checkpoint_directory(
    const std::filesystem::path& staging_directory,
    const std::filesystem::path& target_directory);
void remove_checkpoint_directory(const std::filesystem::path& directory) noexcept;

}  // namespace pulsar
