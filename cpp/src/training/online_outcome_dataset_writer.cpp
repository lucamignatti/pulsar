#include "pulsar/training/online_outcome_dataset_writer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pulsar {

OnlineOutcomeDatasetWriter::OnlineOutcomeDatasetWriter(std::filesystem::path output_root, int obs_dim)
    : output_root_(std::move(output_root)), obs_dim_(obs_dim) {
  if (obs_dim_ <= 0) {
    throw std::invalid_argument("OnlineOutcomeDatasetWriter requires a positive obs_dim.");
  }
  std::filesystem::create_directories(output_root_ / "trajectories");
}

void OnlineOutcomeDatasetWriter::record_trajectory(const torch::Tensor& obs_cpu, std::int64_t outcome) {
  if (obs_cpu.device().type() != torch::kCPU || obs_cpu.dim() != 2 || obs_cpu.size(1) != obs_dim_) {
    throw std::invalid_argument("OnlineOutcomeDatasetWriter received an invalid trajectory tensor.");
  }
  const std::filesystem::path path =
      output_root_ / "trajectories" / ("trajectory_" + std::to_string(static_cast<long long>(trajectories_written_)) + ".pt");
  torch::save(obs_cpu.contiguous(), path);
  std::ofstream label(output_root_ / "trajectories" / ("trajectory_" + std::to_string(static_cast<long long>(trajectories_written_)) + ".json"));
  label << nlohmann::json{{"outcome", outcome}, {"samples", obs_cpu.size(0)}}.dump(2) << '\n';
  samples_written_ += obs_cpu.size(0);
  trajectories_written_ += 1;
}

void OnlineOutcomeDatasetWriter::finish() {
  std::filesystem::create_directories(output_root_);
  std::ofstream metadata(output_root_ / "metadata.json");
  metadata << nlohmann::json{
      {"schema_version", 4},
      {"samples_written", samples_written_},
      {"trajectories_written", trajectories_written_},
  }.dump(2) << '\n';
}

std::int64_t OnlineOutcomeDatasetWriter::samples_written() const {
  return samples_written_;
}

std::int64_t OnlineOutcomeDatasetWriter::trajectories_written() const {
  return trajectories_written_;
}

}  // namespace pulsar

#endif
