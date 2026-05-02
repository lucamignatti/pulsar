#include "pulsar/training/bc_pretrainer.hpp"

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

namespace pulsar {
namespace {

void append_metrics_line(
    const std::filesystem::path& output_dir,
    int epoch_index,
    const char* phase,
    const BCEpochMetrics& metrics) {
  nlohmann::json line = {
      {"epoch", epoch_index},
      {"phase", phase},
      {"behavior_loss", metrics.behavior_loss},
      {"behavior_accuracy", metrics.behavior_accuracy},
      {"value_loss", metrics.value_loss},
      {"behavior_samples", metrics.behavior_samples},
      {"value_samples", metrics.value_samples},
      {"samples", metrics.samples},
  };
  std::filesystem::create_directories(output_dir);
  std::ofstream output(output_dir / "bc_metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

BCEpochMetrics average_metrics(BCEpochMetrics metrics) {
  if (metrics.behavior_samples > 0) {
    metrics.behavior_loss /= static_cast<double>(metrics.behavior_samples);
    metrics.behavior_accuracy /= static_cast<double>(metrics.behavior_samples);
  }
  if (metrics.value_samples > 0) {
    metrics.value_loss /= static_cast<double>(metrics.value_samples);
  }
  return metrics;
}

void configure_cuda_runtime(const torch::Device& device) {
  if (!device.is_cuda()) {
    return;
  }
  at::globalContext().setAllowTF32CuBLAS(true);
  at::globalContext().setAllowTF32CuDNN(true);
}

}  // namespace

BCPretrainer::BCPretrainer(ExperimentConfig config)
    : config_(std::move(config)),
      train_dataset_(config_.offline_dataset.train_manifest),
      val_dataset_(
          config_.offline_dataset.val_manifest.empty() ? config_.offline_dataset.train_manifest
                                                        : config_.offline_dataset.val_manifest),
      actor_(PPOActor(config_.model)),
      actor_normalizer_(config_.model.observation_dim),
      actor_optimizer_(std::make_unique<torch::optim::AdamW>(
          actor_->parameters(),
          torch::optim::AdamWOptions(config_.behavior_cloning.learning_rate)
              .weight_decay(config_.behavior_cloning.weight_decay))),
      device_(config_.ppo.device) {
  configure_cuda_runtime(device_);
  validate_config();
  actor_->to(device_);
  actor_normalizer_.to(device_);
}

void BCPretrainer::validate_config() const {
  if (train_dataset_.empty()) {
    throw std::runtime_error("BC pretraining requires a non-empty training manifest.");
  }
  if (train_dataset_.observation_dim() != config_.model.observation_dim) {
    throw std::runtime_error("Offline manifest observation_dim does not match model.observation_dim.");
  }
  if (train_dataset_.action_dim() != config_.model.action_dim) {
    throw std::runtime_error("Offline manifest action_dim does not match model.action_dim.");
  }
  if (!train_dataset_.has_episode_starts()) {
    throw std::runtime_error("BC pretraining requires trajectory-safe manifests with episode_starts_path.");
  }
}

void BCPretrainer::fit_normalizers() {
  train_dataset_.for_each_batch(
      config_.offline_dataset.batch_size,
      false,
      config_.offline_dataset.seed,
      [&](const OfflineTensorBatch& batch) {
        torch::NoGradGuard no_grad;
        actor_normalizer_.update(batch.obs.to(device_));
      });
}

torch::Tensor BCPretrainer::map_outcome_to_value_target(const torch::Tensor& outcomes) const {
  torch::Tensor targets = torch::zeros_like(outcomes, torch::TensorOptions().dtype(torch::kFloat32));
  targets.masked_fill_(outcomes == 0, config_.outcome.score);
  targets.masked_fill_(outcomes == 1, config_.outcome.concede);
  return targets;
}

BCEpochMetrics BCPretrainer::train_epoch(int epoch_index) {
  BCEpochMetrics metrics{};
  actor_->train();
  train_dataset_.for_each_batch(
      config_.offline_dataset.batch_size,
      config_.offline_dataset.shuffle,
      config_.offline_dataset.seed + static_cast<std::uint64_t>(epoch_index),
      [&](const OfflineTensorBatch& batch) {
        const torch::Tensor obs = batch.obs.to(device_);
        const torch::Tensor normalized =
            actor_normalizer_
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

        torch::Tensor behavior_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        const torch::Tensor flat_logits =
            output.policy_logits.reshape({-1, config_.model.action_dim}).index({flat_valid});
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
            if (config_.behavior_cloning.label_smoothing > 0.0F) {
              options = options.label_smoothing(config_.behavior_cloning.label_smoothing);
            }
            behavior_loss = torch::nn::functional::cross_entropy(flat_logits, target_actions, options);
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(target_actions).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
          }
          metrics.behavior_loss += behavior_loss.item<double>() * static_cast<double>(flat_logits.size(0));
          metrics.behavior_samples += flat_logits.size(0);
        }

        torch::Tensor value_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        if (batch.outcomes.defined()) {
          const torch::Tensor flat_outcomes = batch.outcomes.reshape({-1}).index({flat_valid});
          const torch::Tensor value_targets = map_outcome_to_value_target(flat_outcomes);
          const torch::Tensor flat_value_logits =
              output.value_logits.reshape({-1, config_.model.value_num_atoms}).index({flat_valid});
          const torch::Tensor atom_support = actor_->value_support().to(device_);
          const torch::Tensor proj = torch::log_softmax(flat_value_logits, -1);
          torch::Tensor proj_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
          const float delta_z = (config_.model.value_v_max - config_.model.value_v_min) /
                               static_cast<float>(config_.model.value_num_atoms - 1);
          const torch::Tensor clamped = value_targets.clamp(config_.model.value_v_min, config_.model.value_v_max);
          const torch::Tensor b = (clamped - config_.model.value_v_min) / delta_z;
          const torch::Tensor lower = b.floor().clamp(0, config_.model.value_num_atoms - 1).to(torch::kLong);
          const torch::Tensor upper = b.ceil().clamp(0, config_.model.value_num_atoms - 1).to(torch::kLong);
          const torch::Tensor weight_upper = (b - lower.to(torch::kFloat32)).clamp(0.0, 1.0);
          const torch::Tensor lower_probs = proj.gather(-1, lower.unsqueeze(-1)).squeeze(-1);
          const torch::Tensor upper_probs = proj.gather(-1, upper.unsqueeze(-1)).squeeze(-1);
          proj_loss = -(lower_probs * (1.0 - weight_upper) + upper_probs * weight_upper).mean();
          value_loss = proj_loss;
          metrics.value_loss += value_loss.item<double>() * static_cast<double>(flat_logits.size(0));
          metrics.value_samples += flat_logits.size(0);
        }

        const torch::Tensor loss = behavior_loss + value_loss;
        if (loss.requires_grad()) {
          loss.backward();
          torch::nn::utils::clip_grad_norm_(actor_->parameters(), config_.behavior_cloning.max_grad_norm);
          actor_optimizer_->step();
        }
        metrics.samples += batch.valid_mask.sum().item<std::int64_t>();
      });
  return average_metrics(metrics);
}

BCEpochMetrics BCPretrainer::evaluate() {
  BCEpochMetrics metrics{};
  actor_->eval();
  torch::NoGradGuard no_grad;
  val_dataset_.for_each_batch(
      config_.offline_dataset.batch_size,
      false,
      config_.offline_dataset.seed,
      [&](const OfflineTensorBatch& batch) {
        const torch::Tensor obs = batch.obs.to(device_);
        const torch::Tensor normalized =
            actor_normalizer_
                .normalize(obs.reshape({-1, config_.model.observation_dim}))
                .reshape_as(obs);
        const torch::Tensor starts = batch.episode_starts.to(device_);
        const torch::Tensor valid = batch.valid_mask.to(device_);
        const torch::Tensor actions = batch.actions.defined() ? batch.actions.to(device_) : torch::Tensor{};
        const torch::Tensor action_probs = batch.action_probs.defined() ? batch.action_probs.to(device_) : torch::Tensor{};
        if (!actions.defined() && !action_probs.defined()) {
          return;
        }
        ContinuumState state = actor_->initial_state(normalized.size(1), device_);
        ActorSequenceOutput output = actor_->forward_sequence(normalized, std::move(state), starts);
        const torch::Tensor flat_valid = valid.reshape({-1});
        const torch::Tensor flat_logits =
            output.policy_logits.reshape({-1, config_.model.action_dim}).index({flat_valid});
        if (flat_logits.size(0) > 0) {
          if (action_probs.defined()) {
            const torch::Tensor targets =
                action_probs.reshape({-1, config_.model.action_dim}).index({flat_valid});
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(targets.argmax(-1)).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
          } else {
            const torch::Tensor target_actions = actions.reshape({-1}).index({flat_valid});
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(target_actions).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
          }
          metrics.behavior_samples += flat_logits.size(0);
          metrics.samples += flat_logits.size(0);
        }
      });
  return average_metrics(metrics);
}

void BCPretrainer::save_checkpoint(const std::string& output_dir, int epoch_index) const {
  namespace fs = std::filesystem;
  const fs::path base(output_dir);
  fs::create_directories(base);
  save_experiment_config(config_, (base / "config.json").string());
  save_checkpoint_metadata(
      CheckpointMetadata{
          .schema_version = config_.schema_version,
          .obs_schema_version = config_.obs_schema_version,
          .config_hash = config_hash(config_),
          .action_table_hash = action_table_hash(config_.action_table),
          .architecture_name = "ppo_continuum",
          .device = config_.ppo.device,
          .global_step = 0,
          .update_index = 0,
      },
      (base / "metadata.json").string());

  torch::serialize::OutputArchive actor_archive;
  actor_->save(actor_archive);
  actor_normalizer_.save(actor_archive);
  actor_archive.save_to((base / "model.pt").string());

  torch::serialize::OutputArchive optimizer_archive;
  actor_optimizer_->save(optimizer_archive);
  optimizer_archive.save_to((base / "actor_optimizer.pt").string());
}

void BCPretrainer::train(const std::string& output_dir, const std::string&) {
  if (!config_.behavior_cloning.enabled) {
    throw std::runtime_error("BC pretraining requires behavior_cloning.enabled = true.");
  }
  fit_normalizers();
  int epoch_index = 0;
  for (int epoch = 1; epoch <= config_.behavior_cloning.epochs; ++epoch) {
    epoch_index += 1;
    const BCEpochMetrics metrics = train_epoch(epoch);
    append_metrics_line(output_dir, epoch_index, "train", metrics);
    std::cout << "epoch=" << epoch
              << " behavior_loss=" << metrics.behavior_loss
              << " behavior_accuracy=" << metrics.behavior_accuracy
              << " value_loss=" << metrics.value_loss
              << '\n';
  }

  const BCEpochMetrics val = evaluate();
  append_metrics_line(output_dir, epoch_index, "val", val);
  std::cout << "phase=val"
            << " behavior_accuracy=" << val.behavior_accuracy
            << '\n';
  save_checkpoint(output_dir, epoch_index);
}

}  // namespace pulsar

#endif
