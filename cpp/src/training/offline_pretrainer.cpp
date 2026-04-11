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
  return ((correct * weights).sum() / weights.sum().clamp_min(1.0e-6)).item<double>();
}

ContinuumState detach_state(ContinuumState state) {
  state.workspace = state.workspace.detach();
  state.stm_keys = state.stm_keys.detach();
  state.stm_values = state.stm_values.detach();
  state.stm_strengths = state.stm_strengths.detach();
  state.stm_write_index = state.stm_write_index.detach();
  state.ltm_coeffs = state.ltm_coeffs.detach();
  state.timestep = state.timestep.detach();
  return state;
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
      policy_model_(SharedActorCritic(config_.model, config_.ppo)),
      normalizer_(config_.model.observation_dim),
      optimizer_(
          policy_model_->parameters(),
          torch::optim::AdamWOptions(std::min(
                                         config_.behavior_cloning.learning_rate,
                                         config_.next_goal_predictor.learning_rate))
              .weight_decay(std::max(
                  config_.behavior_cloning.weight_decay,
                  config_.next_goal_predictor.weight_decay))),
      device_(config_.ppo.device) {
  validate_config();
  policy_model_->to(device_);
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
  if (!train_dataset_.has_episode_starts()) {
    throw std::runtime_error(
        "Offline pretrainer requires trajectory-safe manifests with episode_starts_path entries. "
        "Re-run scripts/preprocess_kaggle_2v2.py with the current repo version.");
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

OfflineEpochMetrics OfflinePretrainer::run_epoch(
    const OfflineTensorDataset& dataset,
    bool training,
    int epoch_index) {
  OfflineEpochMetrics metrics{};
  const bool train_policy = training && config_.behavior_cloning.enabled &&
                            epoch_index <= config_.behavior_cloning.epochs;
  const bool train_ngp = training && config_.next_goal_predictor.enabled &&
                         epoch_index <= config_.next_goal_predictor.epochs;
  const bool shuffle = training ? config_.offline_dataset.shuffle : false;
  const std::uint64_t seed = config_.offline_dataset.seed + static_cast<std::uint64_t>(epoch_index);
  const torch::Tensor ngp_class_weights =
      torch::tensor(config_.next_goal_predictor.class_weights, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_);
  const std::int64_t sequence_length = std::max<std::int64_t>(1, config_.behavior_cloning.sequence_length);

  dataset.for_each_trajectory(
      shuffle,
      seed,
      [&](const OfflineTensorBatch& batch) {
        const torch::Tensor obs = batch.obs.to(device_);
        const torch::Tensor weights = batch.weights.to(device_);
        const torch::Tensor normalized = normalizer_.normalize(obs).contiguous();
        const torch::Tensor starts = batch.episode_starts.to(device_).to(torch::kFloat32).contiguous();
        const std::int64_t total_rows = normalized.size(0);
        if (total_rows <= 0) {
          return;
        }
        ContinuumState state = policy_model_->initial_state(1, device_);
        const torch::Tensor actions = batch.actions.defined() ? batch.actions.to(device_) : torch::Tensor{};
        const torch::Tensor next_goal = batch.next_goal.defined() ? batch.next_goal.to(device_) : torch::Tensor{};

        for (std::int64_t offset = 0; offset < total_rows; offset += sequence_length) {
          const std::int64_t chunk_length = std::min<std::int64_t>(sequence_length, total_rows - offset);
          const torch::Tensor chunk_obs = normalized.narrow(0, offset, chunk_length).unsqueeze(1);
          const torch::Tensor chunk_starts = starts.narrow(0, offset, chunk_length).unsqueeze(1);

          auto run_forward = [&]() {
            return policy_model_->forward_sequence(chunk_obs, state, chunk_starts);
          };

          SequenceOutput output;
          if (training) {
            output = run_forward();
          } else {
            torch::NoGradGuard no_grad;
            output = run_forward();
          }

          torch::Tensor total_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
          const torch::Tensor chunk_weights = weights.narrow(0, offset, chunk_length);

          if (config_.behavior_cloning.enabled && actions.defined()) {
            const torch::Tensor chunk_actions = actions.narrow(0, offset, chunk_length);
            const torch::Tensor logits = output.policy_logits.squeeze(1);
            const torch::Tensor loss = weighted_cross_entropy(
                logits,
                chunk_actions,
                chunk_weights,
                config_.behavior_cloning.label_smoothing);
            metrics.policy_loss += loss.item<double>() * static_cast<double>(chunk_length);
            metrics.policy_accuracy +=
                weighted_accuracy(logits, chunk_actions, chunk_weights) * static_cast<double>(chunk_length);
            total_loss = total_loss + loss;
          }

          if (config_.next_goal_predictor.enabled && next_goal.defined()) {
            const torch::Tensor chunk_labels = next_goal.narrow(0, offset, chunk_length);
            const torch::Tensor logits = output.next_goal_logits.squeeze(1);
            const torch::Tensor loss = weighted_cross_entropy(
                logits,
                chunk_labels,
                chunk_weights,
                config_.next_goal_predictor.label_smoothing,
                ngp_class_weights);
            metrics.ngp_loss += loss.item<double>() * static_cast<double>(chunk_length);
            metrics.ngp_accuracy +=
                weighted_accuracy(logits, chunk_labels, chunk_weights) * static_cast<double>(chunk_length);
            total_loss = total_loss + loss;
          }

          if (training && total_loss.requires_grad()) {
            optimizer_.zero_grad();
            total_loss.backward();
            torch::nn::utils::clip_grad_norm_(
                policy_model_->parameters(),
                std::max(config_.behavior_cloning.max_grad_norm, config_.next_goal_predictor.max_grad_norm));
            optimizer_.step();
            state = detach_state(std::move(output.final_state));
          } else {
            state = detach_state(std::move(output.final_state));
          }

          metrics.samples += chunk_length;
        }
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

OfflineEpochMetrics OfflinePretrainer::run_training_epoch(int epoch_index) {
  return run_epoch(train_dataset_, true, epoch_index);
}

OfflineEpochMetrics OfflinePretrainer::evaluate() {
  return run_epoch(val_dataset_, false, 0);
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
          .architecture_name = "continuum_dppo",
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
  optimizer_.save(optimizer_archive);
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
          .architecture_name = "continuum_dppo",
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
  optimizer_.save(optimizer_archive);
  optimizer_archive.save_to((base / "optimizer.pt").string());
}

void OfflinePretrainer::train(const std::string& output_dir, const std::string& config_path) {
  fit_normalizer();
  WandbLogger wandb(config_.wandb, output_dir, config_path, "offline_pretrain");

  const int max_epochs = std::max(config_.behavior_cloning.epochs, config_.next_goal_predictor.epochs);
  for (int epoch = 1; epoch <= max_epochs; ++epoch) {
    const auto start = std::chrono::steady_clock::now();
    policy_model_->train();
    const OfflineEpochMetrics train_metrics = run_training_epoch(epoch);
    policy_model_->eval();
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
    if (wandb.enabled()) {
      wandb.log(nlohmann::json{
          {"epoch", epoch},
          {"train_policy_loss", train_metrics.policy_loss},
          {"train_ngp_loss", train_metrics.ngp_loss},
          {"train_policy_accuracy", train_metrics.policy_accuracy},
          {"train_ngp_accuracy", train_metrics.ngp_accuracy},
          {"train_samples", train_metrics.samples},
          {"val_policy_loss", val_metrics.policy_loss},
          {"val_ngp_loss", val_metrics.ngp_loss},
          {"val_policy_accuracy", val_metrics.policy_accuracy},
          {"val_ngp_accuracy", val_metrics.ngp_accuracy},
          {"val_samples", val_metrics.samples},
          {"epoch_seconds", seconds},
      });
    }
    if (config_.behavior_cloning.enabled) {
      save_policy_checkpoint(output_dir, epoch);
    }
    if (config_.next_goal_predictor.enabled) {
      save_next_goal_checkpoint(output_dir, epoch);
    }
  }
  wandb.finish();
}

}  // namespace pulsar

#endif
