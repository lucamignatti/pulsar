#include "pulsar/training/bc_pretrainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/training/cuda_utils.hpp"
#include "pulsar/training/ppo_math.hpp"

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
  seed_everything(config_.env.seed);
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
  if (!val_dataset_.has_episode_starts()) {
    throw std::runtime_error("BC pretraining validation manifest must also have episode_starts_path.");
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
  targets.masked_fill_(outcomes == 2, config_.outcome.neutral);
  return targets;
}

BCEpochMetrics BCPretrainer::train_epoch(int epoch_index) {
  BCEpochMetrics metrics{};
  actor_->train();
  train_dataset_.for_each_packed_trajectory_batch(
      config_.offline_dataset.batch_size,
      config_.offline_dataset.shuffle,
      config_.offline_dataset.seed + static_cast<std::uint64_t>(epoch_index),
      [&](const OfflineTensorPackedBatch& batch) {
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
            torch::Tensor flat_weights = torch::ones({flat_logits.size(0)},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            if (batch.weights.defined()) {
              flat_weights = batch.weights.reshape({-1}).to(device_).index({flat_valid});
            }
            torch::Tensor per_sample_kl = -(targets * torch::log_softmax(flat_logits, -1)).sum(-1);
            behavior_loss = (per_sample_kl * flat_weights).sum() / flat_weights.sum().clamp_min(1e-8);
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(targets.argmax(-1)).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
          } else {
            const torch::Tensor target_actions = actions.reshape({-1}).index({flat_valid});
            torch::Tensor flat_weights = torch::ones({flat_logits.size(0)},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            if (batch.weights.defined()) {
              flat_weights = batch.weights.reshape({-1}).to(device_).index({flat_valid});
            }
            auto options = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
            if (config_.behavior_cloning.label_smoothing > 0.0F) {
              options = options.label_smoothing(config_.behavior_cloning.label_smoothing);
            }
            torch::Tensor per_sample_ce = torch::nn::functional::cross_entropy(flat_logits, target_actions, options);
            behavior_loss = (per_sample_ce * flat_weights).sum() / flat_weights.sum().clamp_min(1e-8);
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(target_actions).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
          }
          metrics.behavior_loss += behavior_loss.item<double>() * static_cast<double>(flat_logits.size(0));
          metrics.behavior_samples += flat_logits.size(0);
        }

        torch::Tensor value_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        if (batch.outcome.defined() && batch.outcome_known.defined()) {
          const torch::Tensor flat_outcomes = batch.outcome.reshape({-1}).index({flat_valid});
          const torch::Tensor flat_known = batch.outcome_known.reshape({-1}).to(device_).index({flat_valid}) > 0.5F;
          if (flat_known.sum().item<int64_t>() > 0) {
            const torch::Tensor known_outcomes = flat_outcomes.index({flat_known});
            const torch::Tensor value_targets = map_outcome_to_value_target(known_outcomes);
            const torch::Tensor flat_value_logits_all =
                output.value_ext.logits.reshape({-1, config_.model.value_num_atoms}).index({flat_valid});
            const torch::Tensor known_value_logits = flat_value_logits_all.index({flat_known});
            const torch::Tensor atom_support = actor_->value_support("extrinsic").to(device_);
            value_loss = distributional_value_loss(
                known_value_logits, value_targets,
                config_.model.value_v_min, config_.model.value_v_max, config_.model.value_num_atoms);
            metrics.value_loss += value_loss.item<double>() * static_cast<double>(known_value_logits.size(0));
            metrics.value_samples += known_value_logits.size(0);
          }
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
  val_dataset_.for_each_packed_trajectory_batch(
      config_.offline_dataset.batch_size,
      false,
      config_.offline_dataset.seed,
      [&](const OfflineTensorPackedBatch& batch) {
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
            metrics.behavior_loss +=
                (-(targets * torch::log_softmax(flat_logits, -1)).sum(-1).mean()).item<double>() *
                static_cast<double>(flat_logits.size(0));
          } else {
            const torch::Tensor target_actions = actions.reshape({-1}).index({flat_valid});
            metrics.behavior_accuracy +=
                flat_logits.argmax(-1).eq(target_actions).to(torch::kFloat32).mean().item<double>() *
                static_cast<double>(flat_logits.size(0));
            metrics.behavior_loss +=
                torch::nn::functional::cross_entropy(
                    flat_logits, target_actions,
                    torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean))
                    .item<double>() *
                static_cast<double>(flat_logits.size(0));
          }
          metrics.behavior_samples += flat_logits.size(0);
          metrics.samples += flat_logits.size(0);
        }

        if (batch.outcome.defined() && batch.outcome_known.defined()) {
          const torch::Tensor flat_outcomes = batch.outcome.reshape({-1}).index({flat_valid});
          const torch::Tensor flat_known = batch.outcome_known.reshape({-1}).to(device_).index({flat_valid}) > 0.5F;
          if (flat_known.sum().item<int64_t>() > 0) {
            const torch::Tensor known_outcomes = flat_outcomes.index({flat_known});
            const torch::Tensor value_targets = map_outcome_to_value_target(known_outcomes);
            const torch::Tensor flat_value_logits_all =
                output.value_ext.logits.reshape({-1, config_.model.value_num_atoms}).index({flat_valid});
            const torch::Tensor known_value_logits = flat_value_logits_all.index({flat_known});
            const torch::Tensor atom_support = actor_->value_support("extrinsic").to(device_);
            const torch::Tensor value_loss = distributional_value_loss(
                known_value_logits, value_targets,
                config_.model.value_v_min, config_.model.value_v_max, config_.model.value_num_atoms);
            metrics.value_loss += value_loss.item<double>() * static_cast<double>(known_value_logits.size(0));
            metrics.value_samples += known_value_logits.size(0);
          }
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
          .critic_heads = actor_->enabled_critic_heads(),
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

void BCPretrainer::train(const std::string& output_dir, const std::string& config_path) {
  if (!config_.behavior_cloning.enabled) {
    throw std::runtime_error("BC pretraining requires behavior_cloning.enabled = true.");
  }
  fit_normalizers();
  WandbLogger wandb(config_.wandb, output_dir, config_path, "bc_pretrain");
  int epoch_index = 0;
  for (int epoch = 1; epoch <= config_.behavior_cloning.epochs; ++epoch) {
    epoch_index += 1;
    const BCEpochMetrics metrics = train_epoch(epoch);
    append_metrics_line(output_dir, epoch_index, "train", metrics);
    if (wandb.enabled()) {
      wandb.log(nlohmann::json{
          {"epoch", epoch},
          {"phase", "train"},
          {"behavior_loss", metrics.behavior_loss},
          {"behavior_accuracy", metrics.behavior_accuracy},
          {"value_loss", metrics.value_loss},
      });
    }
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
            << " behavior_loss=" << val.behavior_loss
            << " value_loss=" << val.value_loss
            << '\n';
  save_checkpoint(output_dir, epoch_index);
  wandb.finish();
}

}  // namespace pulsar

#endif
