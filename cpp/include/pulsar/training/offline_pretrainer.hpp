#pragma once

#ifdef PULSAR_HAS_TORCH

#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/training/offline_dataset.hpp"

namespace pulsar {

struct OfflineEpochMetrics {
  double policy_loss = 0.0;
  double ngp_loss = 0.0;
  double policy_accuracy = 0.0;
  double ngp_accuracy = 0.0;
  std::int64_t samples = 0;
};

class OfflinePretrainer {
 public:
  explicit OfflinePretrainer(ExperimentConfig config);

  void train(const std::string& output_dir, const std::string& config_path = "");

 private:
  void validate_config() const;
  void maybe_initialize_from_checkpoint();
  void fit_normalizer();
  OfflineEpochMetrics run_training_epoch(int epoch_index);
  OfflineEpochMetrics evaluate();
  OfflineEpochMetrics run_epoch(const OfflineTensorDataset& dataset, bool training, int epoch_index);
  void save_policy_checkpoint(const std::string& output_dir, int epoch_index) const;
  void save_next_goal_checkpoint(const std::string& output_dir, int epoch_index) const;

  ExperimentConfig config_{};
  OfflineTensorDataset train_dataset_;
  OfflineTensorDataset val_dataset_;
  SharedActorCritic policy_model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::optim::AdamW optimizer_;
  torch::Device device_{torch::kCPU};
};

}  // namespace pulsar

#endif
