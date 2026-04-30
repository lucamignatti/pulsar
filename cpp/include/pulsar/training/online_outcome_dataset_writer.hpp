#pragma once

#ifdef PULSAR_HAS_TORCH

#include <cstdint>
#include <filesystem>
#include <vector>

#include <torch/torch.h>

namespace pulsar {

class OnlineOutcomeDatasetWriter {
 public:
  OnlineOutcomeDatasetWriter(std::filesystem::path output_root, int obs_dim);

  void record_trajectory(const torch::Tensor& obs_cpu, std::int64_t outcome);
  void finish();

  [[nodiscard]] std::int64_t samples_written() const;
  [[nodiscard]] std::int64_t trajectories_written() const;

 private:
  std::filesystem::path output_root_{};
  int obs_dim_ = 0;
  std::int64_t samples_written_ = 0;
  std::int64_t trajectories_written_ = 0;
};

}  // namespace pulsar

#endif
