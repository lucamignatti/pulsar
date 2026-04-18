#pragma once

#include <string>

#include "pulsar/config/config.hpp"

namespace pulsar {

void save_checkpoint_metadata(const CheckpointMetadata& metadata, const std::string& path);
CheckpointMetadata load_checkpoint_metadata(const std::string& path);
void validate_checkpoint_metadata(const CheckpointMetadata& metadata, const ExperimentConfig& config);
void validate_inference_checkpoint_metadata(const CheckpointMetadata& metadata, const ExperimentConfig& config);

}  // namespace pulsar
