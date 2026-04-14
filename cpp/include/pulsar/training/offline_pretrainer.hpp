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
  double value_loss = 0.0;
  double policy_accuracy = 0.0;
  double ngp_accuracy = 0.0;
  std::int64_t samples = 0;
  std::int64_t policy_samples = 0;
  std::int64_t ngp_samples = 0;
  std::int64_t value_samples = 0;
};

class OfflinePretrainer {
 public:
  explicit OfflinePretrainer(ExperimentConfig config);

  void train(const std::string& output_dir, const std::string& config_path = "");

 private:
  void validate_config() const;
  void maybe_initialize_from_checkpoint();
  void fit_normalizer();
  OfflineEpochMetrics run_training_epoch(
      int epoch_index,
      SharedActorCritic target_model,
      const ObservationNormalizer& target_normalizer);
  OfflineEpochMetrics evaluate(
      SharedActorCritic target_model,
      const ObservationNormalizer& target_normalizer);
  OfflineEpochMetrics run_epoch(
      const OfflineTensorDataset& dataset,
      bool training,
      int epoch_index,
      SharedActorCritic target_model,
      const ObservationNormalizer& target_normalizer);
  void save_checkpoint(const std::string& output_dir, int epoch_index) const;

  ExperimentConfig config_{};
  OfflineTensorDataset train_dataset_;
  OfflineTensorDataset val_dataset_;
  SharedActorCritic policy_model_{nullptr};
  ObservationNormalizer normalizer_;
  std::vector<torch::Tensor> trunk_parameters_{};
  std::vector<torch::Tensor> policy_head_parameters_{};
  std::vector<torch::Tensor> value_head_parameters_{};
  std::vector<torch::Tensor> ngp_head_parameters_{};
  torch::optim::AdamW trunk_optimizer_;
  torch::optim::AdamW policy_head_optimizer_;
  torch::optim::AdamW value_head_optimizer_;
  torch::optim::AdamW ngp_head_optimizer_;
  torch::Device device_{torch::kCPU};
};

}  // namespace pulsar

#endif
