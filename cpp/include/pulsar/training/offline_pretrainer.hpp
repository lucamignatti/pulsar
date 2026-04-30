#pragma once

#ifdef PULSAR_HAS_TORCH

#include <functional>
#include <memory>
#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/future_evaluator.hpp"
#include "pulsar/model/latent_future_actor.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/training/offline_dataset.hpp"

namespace pulsar {

struct OfflineEpochMetrics {
  double evaluator_loss = 0.0;
  double evaluator_outcome_loss = 0.0;
  double evaluator_delta_loss = 0.0;
  double evaluator_accuracy = 0.0;
  double behavior_loss = 0.0;
  double behavior_accuracy = 0.0;
  double latent_loss = 0.0;
  std::int64_t evaluator_samples = 0;
  std::int64_t evaluator_outcome_samples = 0;
  std::int64_t evaluator_delta_samples = 0;
  std::int64_t behavior_samples = 0;
  std::int64_t latent_samples = 0;
  std::int64_t samples = 0;
};

class OfflinePretrainer {
 public:
  explicit OfflinePretrainer(ExperimentConfig config);

  void train(const std::string& output_dir, const std::string& config_path = "");

 private:
  void validate_config() const;
  void fit_normalizers();
  OfflineEpochMetrics train_evaluator_epoch(int epoch_index);
  OfflineEpochMetrics train_actor_epoch(int epoch_index);
  OfflineEpochMetrics evaluate();
  void save_checkpoint(const std::string& output_dir, int epoch_index) const;

  ExperimentConfig config_{};
  OfflineTensorDataset train_dataset_;
  OfflineTensorDataset val_dataset_;
  LatentFutureActor actor_{nullptr};
  FutureEvaluator evaluator_{nullptr};
  FutureEvaluator target_evaluator_{nullptr};
  ObservationNormalizer actor_normalizer_;
  ObservationNormalizer evaluator_normalizer_;
  std::unique_ptr<torch::optim::AdamW> actor_optimizer_{};
  std::unique_ptr<torch::optim::AdamW> evaluator_optimizer_{};
  torch::Device device_{torch::kCPU};
};

}  // namespace pulsar

#endif
