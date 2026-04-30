#include "pulsar/training/offline_pretrainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <ATen/Context.h>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/training/lfpo_math.hpp"

namespace pulsar {
namespace {

struct FutureWindowBatch {
  torch::Tensor windows;
  torch::Tensor labels;
  torch::Tensor weights;
  torch::Tensor horizon_mask;
  torch::Tensor time_indices;
  torch::Tensor column_indices;
  torch::Tensor actions;
  torch::Tensor action_probs;
};

int max_horizon(const FutureEvaluatorConfig& config) {
  return *std::max_element(config.horizons.begin(), config.horizons.end());
}

FutureWindowBatch build_future_windows(
    const OfflineTensorPackedBatch& batch,
    const FutureEvaluatorConfig& evaluator_config) {
  const int max_h = max_horizon(evaluator_config);
  const int horizon_count = static_cast<int>(evaluator_config.horizons.size());
  const auto obs_dim = batch.obs.size(2);

  std::vector<std::pair<std::int64_t, std::int64_t>> rows;
  for (std::int64_t column = 0; column < static_cast<std::int64_t>(batch.lengths.size()); ++column) {
    const std::int64_t length = batch.lengths[static_cast<std::size_t>(column)];
    for (std::int64_t t = 0; t < length; ++t) {
      if (!batch.outcome.defined() || batch.outcome_known[t][column].item<float>() <= 0.5F) {
        continue;
      }
      rows.emplace_back(t, column);
    }
  }

  const auto row_count = static_cast<std::int64_t>(rows.size());
  FutureWindowBatch out;
  out.windows = torch::zeros({row_count, max_h + 1, obs_dim}, batch.obs.options());
  out.labels = torch::zeros({row_count}, torch::TensorOptions().dtype(torch::kLong));
  out.weights = torch::ones({row_count}, torch::TensorOptions().dtype(torch::kFloat32));
  out.horizon_mask = torch::zeros({row_count, horizon_count}, torch::TensorOptions().dtype(torch::kBool));
  out.time_indices = torch::zeros({row_count}, torch::TensorOptions().dtype(torch::kLong));
  out.column_indices = torch::zeros({row_count}, torch::TensorOptions().dtype(torch::kLong));
  if (batch.actions.defined()) {
    out.actions = torch::zeros({row_count}, batch.actions.options());
  }
  if (batch.action_probs.defined()) {
    out.action_probs = torch::zeros({row_count, batch.action_probs.size(2)}, batch.action_probs.options());
  }

  for (std::int64_t row = 0; row < row_count; ++row) {
    const auto [t, column] = rows[static_cast<std::size_t>(row)];
    const std::int64_t length = batch.lengths[static_cast<std::size_t>(column)];
    const std::int64_t label = batch.outcome[t][column].item<std::int64_t>();
    out.labels[row] = label;
    out.weights[row] = batch.weights[t][column].item<float>();
    out.time_indices[row] = t;
    out.column_indices[row] = column;
    for (int dt = 0; dt <= max_h; ++dt) {
      const std::int64_t src_t = std::min<std::int64_t>(t + dt, length - 1);
      out.windows[row][dt].copy_(batch.obs[src_t][column]);
    }
    for (int h = 0; h < horizon_count; ++h) {
      const bool enough_context = t + evaluator_config.horizons[static_cast<std::size_t>(h)] < length;
      const bool terminal_known = label == 0 || label == 1;
      if (terminal_known || enough_context) {
        out.horizon_mask[row][h] = true;
      }
    }
    if (out.actions.defined()) {
      out.actions[row] = batch.actions[t][column].item<std::int64_t>();
    }
    if (out.action_probs.defined()) {
      out.action_probs[row].copy_(batch.action_probs[t][column]);
    }
  }
  return out;
}

torch::Tensor masked_outcome_loss(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& weights,
    const torch::Tensor& horizon_mask,
    const torch::Tensor& class_weights,
    float label_smoothing) {
  if (logits.numel() == 0 || horizon_mask.sum().item<std::int64_t>() == 0) {
    return torch::zeros({}, logits.options());
  }
  const auto horizon_count = logits.size(1);
  const torch::Tensor flat_logits = logits.reshape({-1, logits.size(2)});
  const torch::Tensor flat_labels = labels.unsqueeze(1).expand({labels.size(0), horizon_count}).reshape({-1});
  const torch::Tensor flat_weights = weights.unsqueeze(1).expand({weights.size(0), horizon_count}).reshape({-1});
  const torch::Tensor flat_mask = horizon_mask.reshape({-1});
  auto options = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
  if (label_smoothing > 0.0F) {
    options = options.label_smoothing(label_smoothing);
  }
  if (class_weights.defined()) {
    options = options.weight(class_weights);
  }
  const torch::Tensor per = torch::nn::functional::cross_entropy(
      flat_logits.index({flat_mask}),
      flat_labels.index({flat_mask}),
      options);
  const torch::Tensor active_weights = flat_weights.index({flat_mask});
  return (per * (active_weights / active_weights.mean().clamp_min(1.0e-6))).mean();
}

double masked_accuracy(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& horizon_mask) {
  if (logits.numel() == 0 || horizon_mask.sum().item<std::int64_t>() == 0) {
    return 0.0;
  }
  const auto horizon_count = logits.size(1);
  const torch::Tensor flat_logits = logits.reshape({-1, logits.size(2)});
  const torch::Tensor flat_labels = labels.unsqueeze(1).expand({labels.size(0), horizon_count}).reshape({-1});
  const torch::Tensor flat_mask = horizon_mask.reshape({-1});
  const torch::Tensor pred = flat_logits.index({flat_mask}).argmax(-1);
  return pred.eq(flat_labels.index({flat_mask})).to(torch::kFloat32).mean().item<double>();
}

void append_metrics_line(
    const std::filesystem::path& output_dir,
    int epoch_index,
    const char* phase,
    const OfflineEpochMetrics& metrics) {
  nlohmann::json line = {
      {"epoch", epoch_index},
      {"phase", phase},
      {"evaluator_loss", metrics.evaluator_loss},
      {"evaluator_accuracy", metrics.evaluator_accuracy},
      {"behavior_loss", metrics.behavior_loss},
      {"behavior_accuracy", metrics.behavior_accuracy},
      {"latent_loss", metrics.latent_loss},
      {"samples", metrics.samples},
      {"evaluator_samples", metrics.evaluator_samples},
      {"behavior_samples", metrics.behavior_samples},
      {"latent_samples", metrics.latent_samples},
  };
  std::filesystem::create_directories(output_dir);
  std::ofstream output(output_dir / "offline_metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

OfflineEpochMetrics average_metrics(OfflineEpochMetrics metrics) {
  if (metrics.evaluator_samples > 0) {
    metrics.evaluator_loss /= static_cast<double>(metrics.evaluator_samples);
    metrics.evaluator_accuracy /= static_cast<double>(metrics.evaluator_samples);
  }
  if (metrics.behavior_samples > 0) {
    metrics.behavior_loss /= static_cast<double>(metrics.behavior_samples);
    metrics.behavior_accuracy /= static_cast<double>(metrics.behavior_samples);
  }
  if (metrics.latent_samples > 0) {
    metrics.latent_loss /= static_cast<double>(metrics.latent_samples);
  }
  return metrics;
}

void configure_cuda_runtime_for_h100(const torch::Device& device) {
  if (!device.is_cuda()) {
    return;
  }
  at::globalContext().setAllowTF32CuBLAS(true);
  at::globalContext().setAllowTF32CuDNN(true);
}

}  // namespace

OfflinePretrainer::OfflinePretrainer(ExperimentConfig config)
    : config_(std::move(config)),
      train_dataset_(config_.offline_dataset.train_manifest),
      val_dataset_(
          config_.offline_dataset.val_manifest.empty() ? config_.offline_dataset.train_manifest
                                                       : config_.offline_dataset.val_manifest),
      actor_(LatentFutureActor(config_.model)),
      evaluator_(FutureEvaluator(config_.future_evaluator, config_.model.observation_dim)),
      target_evaluator_(FutureEvaluator(config_.future_evaluator, config_.model.observation_dim)),
      actor_normalizer_(config_.model.observation_dim),
      evaluator_normalizer_(config_.model.observation_dim),
      actor_optimizer_(std::make_unique<torch::optim::AdamW>(
          actor_->parameters(),
          torch::optim::AdamWOptions(config_.offline_pretraining.actor_learning_rate)
              .weight_decay(config_.offline_pretraining.weight_decay))),
      evaluator_optimizer_(std::make_unique<torch::optim::AdamW>(
          evaluator_->parameters(),
          torch::optim::AdamWOptions(config_.offline_pretraining.evaluator_learning_rate)
              .weight_decay(config_.future_evaluator.weight_decay))),
      device_(config_.lfpo.device) {
  configure_cuda_runtime_for_h100(device_);
  validate_config();
  actor_->to(device_);
  evaluator_->to(device_);
  target_evaluator_->to(device_);
  actor_normalizer_.to(device_);
  evaluator_normalizer_.to(device_);
}

void OfflinePretrainer::validate_config() const {
  if (train_dataset_.empty()) {
    throw std::runtime_error("LFPO pretraining requires a non-empty training manifest.");
  }
  if (train_dataset_.observation_dim() != config_.model.observation_dim) {
    throw std::runtime_error("Offline manifest observation_dim does not match model.observation_dim.");
  }
  if (train_dataset_.action_dim() != config_.model.action_dim) {
    throw std::runtime_error("Offline manifest action_dim does not match model.action_dim.");
  }
  if (train_dataset_.outcome_classes() != config_.future_evaluator.outcome_classes) {
    throw std::runtime_error("Offline manifest outcome_classes does not match future_evaluator.outcome_classes.");
  }
  if (!train_dataset_.has_episode_starts()) {
    throw std::runtime_error("LFPO pretraining requires trajectory-safe manifests with episode_starts_path.");
  }
}

void OfflinePretrainer::fit_normalizers() {
  train_dataset_.for_each_batch(
      config_.offline_dataset.batch_size,
      false,
      config_.offline_dataset.seed,
      [&](const OfflineTensorBatch& batch) {
        torch::NoGradGuard no_grad;
        const torch::Tensor obs = batch.obs.to(device_);
        actor_normalizer_.update(obs);
        evaluator_normalizer_.update(obs);
      });
}

OfflineEpochMetrics OfflinePretrainer::train_evaluator_epoch(int epoch_index) {
  OfflineEpochMetrics metrics{};
  evaluator_->train();
  const torch::Tensor class_weights =
      torch::tensor(config_.future_evaluator.class_weights, torch::TensorOptions().dtype(torch::kFloat32)).to(device_);
  train_dataset_.for_each_packed_trajectory_batch(
      config_.offline_dataset.batch_size,
      config_.offline_dataset.shuffle,
      config_.offline_dataset.seed + static_cast<std::uint64_t>(epoch_index),
      [&](const OfflineTensorPackedBatch& batch) {
        const FutureWindowBatch windows_cpu = build_future_windows(batch, config_.future_evaluator);
        if (windows_cpu.windows.size(0) == 0) {
          return;
        }
        const torch::Tensor windows = evaluator_normalizer_
                                          .normalize(windows_cpu.windows.to(device_).reshape({-1, config_.model.observation_dim}))
                                          .reshape_as(windows_cpu.windows.to(device_));
        const torch::Tensor labels = windows_cpu.labels.to(device_);
        const torch::Tensor weights = windows_cpu.weights.to(device_);
        const torch::Tensor horizon_mask = windows_cpu.horizon_mask.to(device_);
        evaluator_optimizer_->zero_grad();
        const FutureEvaluationOutput output = evaluator_->forward_windows(windows);
        const torch::Tensor loss = masked_outcome_loss(
            output.outcome_logits,
            labels,
            weights,
            horizon_mask,
            class_weights,
            config_.offline_pretraining.label_smoothing);
        loss.backward();
        torch::nn::utils::clip_grad_norm_(evaluator_->parameters(), config_.future_evaluator.max_grad_norm);
        evaluator_optimizer_->step();
        const auto active = horizon_mask.sum().item<std::int64_t>();
        metrics.evaluator_loss += loss.item<double>() * static_cast<double>(active);
        metrics.evaluator_accuracy += masked_accuracy(output.outcome_logits.detach(), labels, horizon_mask) *
                                      static_cast<double>(active);
        metrics.evaluator_samples += active;
        metrics.samples += windows_cpu.windows.size(0);
      });
  return average_metrics(metrics);
}

OfflineEpochMetrics OfflinePretrainer::train_actor_epoch(int epoch_index) {
  OfflineEpochMetrics metrics{};
  actor_->train();
  target_evaluator_->eval();
  train_dataset_.for_each_packed_trajectory_batch(
      config_.offline_dataset.batch_size,
      config_.offline_dataset.shuffle,
      config_.offline_dataset.seed + static_cast<std::uint64_t>(epoch_index),
      [&](const OfflineTensorPackedBatch& batch) {
        const FutureWindowBatch windows_cpu = build_future_windows(batch, config_.future_evaluator);
        const torch::Tensor obs = batch.obs.to(device_);
        const torch::Tensor normalized = actor_normalizer_
                                             .normalize(obs.reshape({-1, config_.model.observation_dim}))
                                             .reshape_as(obs);
        const torch::Tensor starts = batch.episode_starts.to(device_);
        const torch::Tensor valid = batch.valid_mask.to(device_);
        const torch::Tensor actions = batch.actions.defined() ? batch.actions.to(device_) : torch::Tensor{};
        const torch::Tensor action_probs = batch.action_probs.defined() ? batch.action_probs.to(device_) : torch::Tensor{};
        if (!actions.defined() && !action_probs.defined()) {
          return;
        }

        actor_optimizer_->zero_grad();
        ContinuumState state = actor_->initial_state(normalized.size(1), device_);
        ActorSequenceOutput output = actor_->forward_sequence(normalized, std::move(state), starts);
        const torch::Tensor flat_valid = valid.reshape({-1});
        const torch::Tensor flat_logits =
            output.policy_logits.reshape({-1, config_.model.action_dim}).index({flat_valid});
        torch::Tensor behavior_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        if (flat_logits.size(0) > 0) {
          if (action_probs.defined()) {
            const torch::Tensor targets =
                action_probs.reshape({-1, config_.model.action_dim}).index({flat_valid});
            behavior_loss = -(targets * torch::log_softmax(flat_logits, -1)).sum(-1).mean();
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(targets.argmax(-1)).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
          } else {
            const torch::Tensor target_actions = actions.reshape({-1}).index({flat_valid});
            auto options = torch::nn::functional::CrossEntropyFuncOptions();
            if (config_.offline_pretraining.label_smoothing > 0.0F) {
              options = options.label_smoothing(config_.offline_pretraining.label_smoothing);
            }
            behavior_loss = torch::nn::functional::cross_entropy(flat_logits, target_actions, options);
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(target_actions).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
          }
          metrics.behavior_loss += behavior_loss.item<double>() * static_cast<double>(flat_logits.size(0));
          metrics.behavior_samples += flat_logits.size(0);
        }

        torch::Tensor latent_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        if (windows_cpu.windows.size(0) > 0 && windows_cpu.actions.defined()) {
          torch::Tensor target_embeddings;
          {
            torch::NoGradGuard no_grad;
            const torch::Tensor eval_windows =
                evaluator_normalizer_
                    .normalize(windows_cpu.windows.to(device_).reshape({-1, config_.model.observation_dim}))
                    .reshape_as(windows_cpu.windows.to(device_));
            const FutureEvaluationOutput target = target_evaluator_->forward_windows(eval_windows);
            target_embeddings = target.embeddings.detach();
          }
          const torch::Tensor time_idx = windows_cpu.time_indices.to(device_);
          const torch::Tensor col_idx = windows_cpu.column_indices.to(device_);
          const torch::Tensor selected_features = output.features.index({time_idx, col_idx});
          const torch::Tensor pred = actor_->predict_future_latents(selected_features, windows_cpu.actions.to(device_));
          const torch::Tensor mask = windows_cpu.horizon_mask.to(device_).unsqueeze(-1).to(torch::kFloat32);
          const torch::Tensor diff = (pred - target_embeddings).pow(2) * mask;
          latent_loss = diff.sum() / mask.sum().clamp_min(1.0).mul(config_.future_evaluator.latent_dim);
          const auto active = windows_cpu.horizon_mask.sum().item<std::int64_t>();
          metrics.latent_loss += latent_loss.item<double>() * static_cast<double>(active);
          metrics.latent_samples += active;
        }

        const torch::Tensor loss =
            config_.offline_pretraining.behavior_cloning_loss_coef * behavior_loss +
            config_.offline_pretraining.latent_loss_coef * latent_loss;
        if (loss.requires_grad()) {
          loss.backward();
          torch::nn::utils::clip_grad_norm_(actor_->parameters(), config_.offline_pretraining.max_grad_norm);
          actor_optimizer_->step();
        }
        metrics.samples += batch.valid_mask.sum().item<std::int64_t>();
      });
  return average_metrics(metrics);
}

OfflineEpochMetrics OfflinePretrainer::evaluate() {
  OfflineEpochMetrics metrics{};
  evaluator_->eval();
  torch::NoGradGuard no_grad;
  val_dataset_.for_each_packed_trajectory_batch(
      config_.offline_dataset.batch_size,
      false,
      config_.offline_dataset.seed,
      [&](const OfflineTensorPackedBatch& batch) {
        const FutureWindowBatch windows_cpu = build_future_windows(batch, config_.future_evaluator);
        if (windows_cpu.windows.size(0) == 0) {
          return;
        }
        const torch::Tensor windows = evaluator_normalizer_
                                          .normalize(windows_cpu.windows.to(device_).reshape({-1, config_.model.observation_dim}))
                                          .reshape_as(windows_cpu.windows.to(device_));
        const torch::Tensor labels = windows_cpu.labels.to(device_);
        const torch::Tensor horizon_mask = windows_cpu.horizon_mask.to(device_);
        const FutureEvaluationOutput output = evaluator_->forward_windows(windows);
        const auto active = horizon_mask.sum().item<std::int64_t>();
        metrics.evaluator_accuracy += masked_accuracy(output.outcome_logits, labels, horizon_mask) *
                                      static_cast<double>(active);
        metrics.evaluator_samples += active;
        metrics.samples += windows_cpu.windows.size(0);
      });
  return average_metrics(metrics);
}

void OfflinePretrainer::save_checkpoint(const std::string& output_dir, int epoch_index) const {
  namespace fs = std::filesystem;
  const fs::path base(output_dir);
  fs::create_directories(base);
  fs::create_directories(base / "future_evaluator");
  save_experiment_config(config_, (base / "config.json").string());
  save_checkpoint_metadata(
      CheckpointMetadata{
          .schema_version = config_.schema_version,
          .obs_schema_version = config_.obs_schema_version,
          .config_hash = config_hash(config_),
          .action_table_hash = action_table_hash(config_.action_table),
          .architecture_name = "lfpo_continuum",
          .device = config_.lfpo.device,
          .global_step = 0,
          .update_index = 0,
          .future_evaluator_checkpoint = "future_evaluator",
          .future_evaluator_config_hash = hash_string(nlohmann::json(config_.future_evaluator).dump()),
          .future_evaluator_global_step = train_dataset_.sample_count(),
          .future_evaluator_update_index = epoch_index,
          .future_evaluator_target_update_index = epoch_index,
      },
      (base / "metadata.json").string());

  torch::serialize::OutputArchive actor_archive;
  actor_->save(actor_archive);
  actor_normalizer_.save(actor_archive);
  actor_archive.save_to((base / "model.pt").string());

  torch::serialize::OutputArchive evaluator_archive;
  evaluator_->save(evaluator_archive);
  evaluator_normalizer_.save(evaluator_archive);
  evaluator_archive.save_to((base / "future_evaluator" / "model.pt").string());

  torch::serialize::OutputArchive actor_optimizer_archive;
  actor_optimizer_->save(actor_optimizer_archive);
  actor_optimizer_archive.save_to((base / "actor_optimizer.pt").string());

  torch::serialize::OutputArchive evaluator_optimizer_archive;
  evaluator_optimizer_->save(evaluator_optimizer_archive);
  evaluator_optimizer_archive.save_to((base / "future_evaluator" / "optimizer.pt").string());
}

void OfflinePretrainer::train(const std::string& output_dir, const std::string&) {
  fit_normalizers();
  int epoch_index = 0;
  for (int epoch = 1; epoch <= config_.offline_pretraining.evaluator_epochs; ++epoch) {
    epoch_index += 1;
    const OfflineEpochMetrics metrics = train_evaluator_epoch(epoch);
    append_metrics_line(output_dir, epoch_index, "future_evaluator", metrics);
    std::cout << "phase=future_evaluator epoch=" << epoch
              << " loss=" << metrics.evaluator_loss
              << " accuracy=" << metrics.evaluator_accuracy
              << '\n';
  }

  target_evaluator_ = clone_future_evaluator(evaluator_, device_);
  target_evaluator_->eval();

  for (int epoch = 1; epoch <= config_.offline_pretraining.actor_epochs; ++epoch) {
    epoch_index += 1;
    const OfflineEpochMetrics metrics = train_actor_epoch(epoch);
    append_metrics_line(output_dir, epoch_index, "actor", metrics);
    std::cout << "phase=actor epoch=" << epoch
              << " behavior_loss=" << metrics.behavior_loss
              << " behavior_accuracy=" << metrics.behavior_accuracy
              << " latent_loss=" << metrics.latent_loss
              << '\n';
  }

  const OfflineEpochMetrics val = evaluate();
  append_metrics_line(output_dir, epoch_index, "val", val);
  save_checkpoint(output_dir, epoch_index);
}

}  // namespace pulsar

#endif
