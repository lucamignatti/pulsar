#include "pulsar/training/offline_pretrainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "pulsar/checkpoint/checkpoint.hpp"

namespace pulsar {
namespace {

torch::Tensor weighted_cross_entropy(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& weights,
    double label_smoothing,
    const torch::Tensor& class_weights = {}) {
  auto options = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
  if (label_smoothing > 0.0) {
    options = options.label_smoothing(label_smoothing);
  }
  if (class_weights.defined()) {
    options = options.weight(class_weights);
  }
  const torch::Tensor per_sample = torch::nn::functional::cross_entropy(logits, labels, options);
  const torch::Tensor normalized_weights = weights / weights.mean().clamp_min(1.0e-6);
  return (per_sample * normalized_weights).mean();
}

double weighted_accuracy(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& weights) {
  const torch::Tensor predictions = logits.argmax(-1);
  const torch::Tensor correct = predictions.eq(labels).to(torch::kFloat32);
  const torch::Tensor weighted = correct * weights;
  return (weighted.sum() / weights.sum().clamp_min(1.0e-6)).item<double>();
}

void append_offline_metrics_line(
    const std::filesystem::path& output_dir,
    int epoch_index,
    const OfflineEpochMetrics& train,
    const OfflineEpochMetrics& val) {
  nlohmann::json line = {
      {"epoch", epoch_index},
      {"train_policy_loss", train.policy_loss},
      {"train_ngp_loss", train.ngp_loss},
      {"train_policy_accuracy", train.policy_accuracy},
      {"train_ngp_accuracy", train.ngp_accuracy},
      {"train_samples", train.samples},
      {"val_policy_loss", val.policy_loss},
      {"val_ngp_loss", val.ngp_loss},
      {"val_policy_accuracy", val.policy_accuracy},
      {"val_ngp_accuracy", val.ngp_accuracy},
      {"val_samples", val.samples},
  };

  std::filesystem::create_directories(output_dir);
  std::ofstream output(output_dir / "offline_metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

}  // namespace

OfflinePretrainer::OfflinePretrainer(ExperimentConfig config)
    : config_(std::move(config)),
      train_dataset_(config_.offline_dataset.train_manifest),
      val_dataset_(
          config_.offline_dataset.val_manifest.empty() ? config_.offline_dataset.train_manifest
                                                       : config_.offline_dataset.val_manifest),
      policy_model_(SharedActorCritic(config_.model)),
      next_goal_model_(NextGoalPredictor(config_.model.observation_dim, config_.next_goal_predictor)),
      normalizer_(config_.model.observation_dim),
      policy_optimizer_(
          policy_model_->parameters(),
          torch::optim::AdamWOptions(config_.behavior_cloning.learning_rate)
              .weight_decay(config_.behavior_cloning.weight_decay)),
      next_goal_optimizer_(
          next_goal_model_->parameters(),
          torch::optim::AdamWOptions(config_.next_goal_predictor.learning_rate)
              .weight_decay(config_.next_goal_predictor.weight_decay)),
      device_(config_.ppo.device) {
  validate_config();
  policy_model_->to(device_);
  next_goal_model_->to(device_);
  normalizer_.to(device_);
}

void OfflinePretrainer::validate_config() const {
  if (train_dataset_.empty()) {
    throw std::runtime_error("Offline pretrainer requires a non-empty training manifest.");
  }
  if (train_dataset_.observation_dim() != config_.model.observation_dim) {
    throw std::runtime_error("Offline manifest observation_dim does not match model.observation_dim.");
  }
  if (train_dataset_.action_dim() != config_.model.action_dim) {
    throw std::runtime_error("Offline manifest action_dim does not match model.action_dim.");
  }
  if (config_.next_goal_predictor.enabled &&
      train_dataset_.next_goal_classes() != config_.next_goal_predictor.num_classes) {
    throw std::runtime_error("Offline manifest next_goal_classes does not match next_goal_predictor.num_classes.");
  }
}

void OfflinePretrainer::fit_normalizer() {
  normalizer_.to(device_);
  train_dataset_.for_each_batch(
      config_.offline_dataset.batch_size,
      false,
      config_.offline_dataset.seed,
      [&](const OfflineTensorBatch& batch) {
        torch::NoGradGuard no_grad;
        normalizer_.update(batch.obs.to(device_));
      });
}

OfflineEpochMetrics OfflinePretrainer::run_training_epoch(int epoch_index) {
  OfflineEpochMetrics metrics{};
  const bool train_policy = config_.behavior_cloning.enabled && epoch_index <= config_.behavior_cloning.epochs;
  const bool train_ngp = config_.next_goal_predictor.enabled && epoch_index <= config_.next_goal_predictor.epochs;
  const torch::Tensor ngp_class_weights =
      torch::tensor(config_.next_goal_predictor.class_weights, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_);

  train_dataset_.for_each_batch(
      config_.offline_dataset.batch_size,
      config_.offline_dataset.shuffle,
      config_.offline_dataset.seed + static_cast<std::uint64_t>(epoch_index),
      [&](const OfflineTensorBatch& batch) {
        const torch::Tensor obs = batch.obs.to(device_);
        const torch::Tensor weights = batch.weights.to(device_);
        const torch::Tensor normalized = normalizer_.normalize(obs);

        if (train_policy) {
          const torch::Tensor actions = batch.actions.to(device_);
          const PolicyOutput output = policy_model_->forward(normalized);
          const torch::Tensor loss = weighted_cross_entropy(
              output.logits,
              actions,
              weights,
              config_.behavior_cloning.label_smoothing);

          policy_optimizer_.zero_grad();
          loss.backward();
          torch::nn::utils::clip_grad_norm_(policy_model_->parameters(), config_.behavior_cloning.max_grad_norm);
          policy_optimizer_.step();

          metrics.policy_loss += loss.item<double>() * static_cast<double>(obs.size(0));
          metrics.policy_accuracy += weighted_accuracy(output.logits, actions, weights) * static_cast<double>(obs.size(0));
        }

        if (train_ngp) {
          const torch::Tensor labels = batch.next_goal.to(device_);
          const torch::Tensor logits = next_goal_model_->forward(normalized);
          const torch::Tensor loss = weighted_cross_entropy(
              logits,
              labels,
              weights,
              config_.next_goal_predictor.label_smoothing,
              ngp_class_weights);

          next_goal_optimizer_.zero_grad();
          loss.backward();
          torch::nn::utils::clip_grad_norm_(
              next_goal_model_->parameters(),
              config_.next_goal_predictor.max_grad_norm);
          next_goal_optimizer_.step();

          metrics.ngp_loss += loss.item<double>() * static_cast<double>(obs.size(0));
          metrics.ngp_accuracy += weighted_accuracy(logits, labels, weights) * static_cast<double>(obs.size(0));
        }

        metrics.samples += obs.size(0);
      });

  if (metrics.samples > 0) {
    if (train_policy) {
      metrics.policy_loss /= static_cast<double>(metrics.samples);
      metrics.policy_accuracy /= static_cast<double>(metrics.samples);
    }
    if (train_ngp) {
      metrics.ngp_loss /= static_cast<double>(metrics.samples);
      metrics.ngp_accuracy /= static_cast<double>(metrics.samples);
    }
  }

  return metrics;
}

OfflineEpochMetrics OfflinePretrainer::evaluate() {
  OfflineEpochMetrics metrics{};
  const torch::Tensor ngp_class_weights =
      torch::tensor(config_.next_goal_predictor.class_weights, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_);
  torch::NoGradGuard no_grad;

  val_dataset_.for_each_batch(
      config_.offline_dataset.val_batch_size,
      false,
      config_.offline_dataset.seed,
      [&](const OfflineTensorBatch& batch) {
        const torch::Tensor obs = batch.obs.to(device_);
        const torch::Tensor weights = batch.weights.to(device_);
        const torch::Tensor normalized = normalizer_.normalize(obs);

        if (config_.behavior_cloning.enabled && batch.actions.defined()) {
          const torch::Tensor actions = batch.actions.to(device_);
          const PolicyOutput output = policy_model_->forward(normalized);
          const torch::Tensor loss = weighted_cross_entropy(
              output.logits,
              actions,
              weights,
              config_.behavior_cloning.label_smoothing);
          metrics.policy_loss += loss.item<double>() * static_cast<double>(obs.size(0));
          metrics.policy_accuracy += weighted_accuracy(output.logits, actions, weights) * static_cast<double>(obs.size(0));
        }

        if (config_.next_goal_predictor.enabled && batch.next_goal.defined()) {
          const torch::Tensor labels = batch.next_goal.to(device_);
          const torch::Tensor logits = next_goal_model_->forward(normalized);
          const torch::Tensor loss = weighted_cross_entropy(
              logits,
              labels,
              weights,
              config_.next_goal_predictor.label_smoothing,
              ngp_class_weights);
          metrics.ngp_loss += loss.item<double>() * static_cast<double>(obs.size(0));
          metrics.ngp_accuracy += weighted_accuracy(logits, labels, weights) * static_cast<double>(obs.size(0));
        }

        metrics.samples += obs.size(0);
      });

  if (metrics.samples > 0) {
    if (config_.behavior_cloning.enabled) {
      metrics.policy_loss /= static_cast<double>(metrics.samples);
      metrics.policy_accuracy /= static_cast<double>(metrics.samples);
    }
    if (config_.next_goal_predictor.enabled) {
      metrics.ngp_loss /= static_cast<double>(metrics.samples);
      metrics.ngp_accuracy /= static_cast<double>(metrics.samples);
    }
  }

  return metrics;
}

void OfflinePretrainer::save_policy_checkpoint(const std::string& output_dir, int epoch_index) const {
  namespace fs = std::filesystem;
  const fs::path base = fs::path(output_dir) / "policy";
  fs::create_directories(base);

  save_experiment_config(config_, (base / "config.json").string());
  save_checkpoint_metadata(
      CheckpointMetadata{
          .schema_version = config_.schema_version,
          .obs_schema_version = config_.obs_schema_version,
          .config_hash = config_hash(config_),
          .action_table_hash = action_table_hash(config_.action_table),
          .architecture_name = "shared_actor_critic",
          .device = config_.ppo.device,
          .global_step = train_dataset_.sample_count(),
          .update_index = epoch_index,
      },
      (base / "metadata.json").string());

  torch::serialize::OutputArchive archive;
  policy_model_->save(archive);
  normalizer_.save(archive);
  archive.save_to((base / "model.pt").string());

  torch::serialize::OutputArchive optimizer_archive;
  policy_optimizer_.save(optimizer_archive);
  optimizer_archive.save_to((base / "optimizer.pt").string());
}

void OfflinePretrainer::save_next_goal_checkpoint(const std::string& output_dir, int epoch_index) const {
  namespace fs = std::filesystem;
  const fs::path base = fs::path(output_dir) / "next_goal";
  fs::create_directories(base);

  save_experiment_config(config_, (base / "config.json").string());
  save_checkpoint_metadata(
      CheckpointMetadata{
          .schema_version = config_.schema_version,
          .obs_schema_version = config_.obs_schema_version,
          .config_hash = config_hash(config_),
          .action_table_hash = action_table_hash(config_.action_table),
          .architecture_name = "next_goal_predictor",
          .device = config_.ppo.device,
          .global_step = train_dataset_.sample_count(),
          .update_index = epoch_index,
      },
      (base / "metadata.json").string());

  torch::serialize::OutputArchive archive;
  next_goal_model_->save(archive);
  normalizer_.save(archive);
  archive.save_to((base / "model.pt").string());

  torch::serialize::OutputArchive optimizer_archive;
  next_goal_optimizer_.save(optimizer_archive);
  optimizer_archive.save_to((base / "optimizer.pt").string());
}

void OfflinePretrainer::train(const std::string& output_dir) {
  fit_normalizer();

  const int max_epochs = std::max(config_.behavior_cloning.epochs, config_.next_goal_predictor.epochs);
  for (int epoch = 1; epoch <= max_epochs; ++epoch) {
    const auto start = std::chrono::steady_clock::now();
    policy_model_->train();
    next_goal_model_->train();
    const OfflineEpochMetrics train_metrics = run_training_epoch(epoch);
    policy_model_->eval();
    next_goal_model_->eval();
    const OfflineEpochMetrics val_metrics = evaluate();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "epoch=" << epoch
              << " seconds=" << seconds
              << " train_policy_loss=" << train_metrics.policy_loss
              << " train_ngp_loss=" << train_metrics.ngp_loss
              << " train_policy_acc=" << train_metrics.policy_accuracy
              << " train_ngp_acc=" << train_metrics.ngp_accuracy
              << " val_policy_loss=" << val_metrics.policy_loss
              << " val_ngp_loss=" << val_metrics.ngp_loss
              << " val_policy_acc=" << val_metrics.policy_accuracy
              << " val_ngp_acc=" << val_metrics.ngp_accuracy
              << '\n';

    append_offline_metrics_line(output_dir, epoch, train_metrics, val_metrics);
    if (config_.behavior_cloning.enabled) {
      save_policy_checkpoint(output_dir, epoch);
    }
    if (config_.next_goal_predictor.enabled) {
      save_next_goal_checkpoint(output_dir, epoch);
    }
  }
}

}  // namespace pulsar

#endif
