#pragma once

#ifdef PULSAR_HAS_TORCH

#include <functional>
#include <memory>
#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/training/offline_dataset.hpp"

namespace pulsar {

struct BCEpochMetrics {
  double behavior_loss = 0.0;
  double behavior_accuracy = 0.0;
  double value_loss = 0.0;
  double forward_loss = 0.0;
  double inverse_loss = 0.0;
  std::int64_t behavior_samples = 0;
  std::int64_t value_samples = 0;
  std::int64_t forward_samples = 0;
  std::int64_t samples = 0;
};

class BCPretrainer {
 public:
  explicit BCPretrainer(ExperimentConfig config);

  void train(const std::string& output_dir, const std::string& config_path = "");

 private:
  void validate_config() const;
  void fit_normalizers();
  BCEpochMetrics train_epoch(int epoch_index);
  BCEpochMetrics evaluate();
  void save_checkpoint(const std::string& output_dir, int epoch_index) const;
  [[nodiscard]] torch::Tensor map_outcome_to_value_target(const torch::Tensor& outcomes) const;

  ExperimentConfig config_{};
  OfflineTensorDataset train_dataset_;
  OfflineTensorDataset val_dataset_;
  PPOActor actor_{nullptr};
  ObservationNormalizer actor_normalizer_;
  std::unique_ptr<torch::optim::AdamW> actor_optimizer_{};
  torch::Device device_{torch::kCPU};
};

}  // namespace pulsar

#endif
