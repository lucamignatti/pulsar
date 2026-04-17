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
#include "pulsar/tracing/tracing.hpp"
#include "pulsar/training/ppo_math.hpp"

namespace pulsar {
namespace {

std::vector<torch::Tensor> collect_head_parameters(
    const SharedActorCritic& model,
    const std::string& prefix) {
  std::vector<torch::Tensor> params;
  for (const auto& item : model->named_parameters(true)) {
    if (item.key().rfind(prefix, 0) == 0) {
      params.push_back(item.value());
    }
  }
  return params;
}

std::vector<torch::Tensor> collect_trunk_parameters(const SharedActorCritic& model) {
  std::vector<torch::Tensor> params;
  for (const auto& item : model->named_parameters(true)) {
    if (item.key().rfind("policy_head.", 0) == 0 ||
        item.key().rfind("value_head.", 0) == 0 ||
        item.key().rfind("next_goal_head.", 0) == 0) {
      continue;
    }
    params.push_back(item.value());
  }
  return params;
}

std::string stable_model_signature(const ExperimentConfig& config) {
  nlohmann::json j = {
      {"model", config.model},
      {"value_num_atoms", config.ppo.value_num_atoms},
      {"value_v_min", config.ppo.value_v_min},
      {"value_v_max", config.ppo.value_v_max},
  };
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
}

void validate_init_checkpoint_compatibility(
    const CheckpointMetadata& metadata,
    const ExperimentConfig& checkpoint_config,
    const ExperimentConfig& active_config,
    const char* checkpoint_name) {
  if (metadata.schema_version != active_config.schema_version) {
    throw std::runtime_error(std::string(checkpoint_name) + " schema_version does not match the active config.");
  }
  if (metadata.obs_schema_version != active_config.obs_schema_version) {
    throw std::runtime_error(std::string(checkpoint_name) + " obs_schema_version does not match the active config.");
  }
  if (metadata.action_table_hash != action_table_hash(active_config.action_table)) {
    throw std::runtime_error(std::string(checkpoint_name) + " action table hash does not match the active config.");
  }
  if (stable_model_signature(checkpoint_config) != stable_model_signature(active_config)) {
    throw std::runtime_error(std::string(checkpoint_name) + " model/value signature does not match the active config.");
  }
}

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

torch::Tensor weighted_soft_cross_entropy(
    const torch::Tensor& logits,
    const torch::Tensor& target_probs,
    const torch::Tensor& weights) {
  const torch::Tensor per_sample = -(target_probs * torch::log_softmax(logits, -1)).sum(-1);
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

double weighted_accuracy_from_probs(
    const torch::Tensor& logits,
    const torch::Tensor& target_probs,
    const torch::Tensor& weights) {
  const torch::Tensor labels = target_probs.argmax(-1);
  return weighted_accuracy(logits, labels, weights);
}

torch::Tensor ngp_scalar(const torch::Tensor& logits) {
  const torch::Tensor probs = torch::softmax(logits, -1);
  return probs.select(-1, 0) - probs.select(-1, 1);
}

torch::Tensor categorical_projection(const ExperimentConfig& config, const torch::Tensor& returns) {
  return categorical_value_projection(
      returns,
      config.ppo.value_v_min,
      config.ppo.value_v_max,
      config.ppo.value_num_atoms);
}

std::int64_t count_valid_rows_in_window(
    const std::vector<std::int64_t>& lengths,
    std::int64_t offset,
    std::int64_t window_length) {
  std::int64_t count = 0;
  for (const std::int64_t length : lengths) {
    if (length <= offset) {
      continue;
    }
    count += std::min<std::int64_t>(window_length, length - offset);
  }
  return count;
}

std::int64_t count_value_rows_in_window(
    const std::vector<std::int64_t>& lengths,
    std::int64_t offset,
    std::int64_t window_length) {
  std::int64_t count = 0;
  for (const std::int64_t length : lengths) {
    const std::int64_t active_length = length - 1;
    if (active_length <= offset) {
      continue;
    }
    count += std::min<std::int64_t>(window_length, active_length - offset);
  }
  return count;
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
      {"train_value_loss", train.value_loss},
      {"train_policy_accuracy", train.policy_accuracy},
      {"train_ngp_accuracy", train.ngp_accuracy},
      {"train_samples", train.samples},
      {"train_policy_samples", train.policy_samples},
      {"train_ngp_samples", train.ngp_samples},
      {"train_value_samples", train.value_samples},
      {"val_policy_loss", val.policy_loss},
      {"val_ngp_loss", val.ngp_loss},
      {"val_value_loss", val.value_loss},
      {"val_policy_accuracy", val.policy_accuracy},
      {"val_ngp_accuracy", val.ngp_accuracy},
      {"val_samples", val.samples},
      {"val_policy_samples", val.policy_samples},
      {"val_ngp_samples", val.ngp_samples},
      {"val_value_samples", val.value_samples},
  };

  std::filesystem::create_directories(output_dir);
  std::ofstream output(output_dir / "offline_metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

void accumulate_offline_benchmark_metrics(
    OfflineBenchmarkMetrics* aggregate,
    const OfflineBenchmarkMetrics& metrics) {
  aggregate->fit_normalizer_seconds += metrics.fit_normalizer_seconds;
  aggregate->train_epoch_seconds += metrics.train_epoch_seconds;
  aggregate->eval_epoch_seconds += metrics.eval_epoch_seconds;
  aggregate->overall_seconds += metrics.overall_seconds;
  aggregate->train_samples += metrics.train_samples;
  aggregate->eval_samples += metrics.eval_samples;
}

void average_offline_benchmark_metrics(OfflineBenchmarkMetrics* aggregate, double count) {
  if (count <= 0.0) {
    return;
  }
  aggregate->fit_normalizer_seconds /= count;
  aggregate->train_epoch_seconds /= count;
  aggregate->eval_epoch_seconds /= count;
  aggregate->overall_seconds /= count;
  aggregate->train_samples = static_cast<std::int64_t>(static_cast<double>(aggregate->train_samples) / count);
  aggregate->eval_samples = static_cast<std::int64_t>(static_cast<double>(aggregate->eval_samples) / count);
}

OfflineEpochMetrics averaged_epoch_metrics(OfflineEpochMetrics metrics) {
  if (metrics.policy_samples > 0) {
    metrics.policy_loss /= static_cast<double>(metrics.policy_samples);
    metrics.policy_accuracy /= static_cast<double>(metrics.policy_samples);
  }
  if (metrics.ngp_samples > 0) {
    metrics.ngp_loss /= static_cast<double>(metrics.ngp_samples);
    metrics.ngp_accuracy /= static_cast<double>(metrics.ngp_samples);
  }
  if (metrics.value_samples > 0) {
    metrics.value_loss /= static_cast<double>(metrics.value_samples);
  }
  return metrics;
}

nlohmann::json make_live_offline_metrics_payload(
    const char* phase,
    int epoch_index,
    const OfflineEpochMetrics& metrics,
    double elapsed_seconds,
    std::int64_t total_samples) {
  const OfflineEpochMetrics averaged = averaged_epoch_metrics(metrics);
  const std::string prefix = std::string("live_") + phase + "_";
  nlohmann::json payload = {
      {"epoch", epoch_index},
      {"live_phase", phase},
      {"live_elapsed_seconds", elapsed_seconds},
      {"live_total_samples", total_samples},
  };
  payload[prefix + "samples"] = metrics.samples;
  payload[prefix + "samples_per_second"] =
      elapsed_seconds > 0.0 ? static_cast<double>(metrics.samples) / elapsed_seconds : 0.0;
  payload[prefix + "progress"] =
      total_samples > 0 ? static_cast<double>(metrics.samples) / static_cast<double>(total_samples) : 0.0;
  payload[prefix + "policy_loss"] = averaged.policy_loss;
  payload[prefix + "ngp_loss"] = averaged.ngp_loss;
  payload[prefix + "value_loss"] = averaged.value_loss;
  payload[prefix + "policy_accuracy"] = averaged.policy_accuracy;
  payload[prefix + "ngp_accuracy"] = averaged.ngp_accuracy;
  payload[prefix + "policy_samples"] = metrics.policy_samples;
  payload[prefix + "ngp_samples"] = metrics.ngp_samples;
  payload[prefix + "value_samples"] = metrics.value_samples;
  return payload;
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
      trunk_parameters_(collect_trunk_parameters(policy_model_)),
      policy_head_parameters_(collect_head_parameters(policy_model_, "policy_head.")),
      value_head_parameters_(collect_head_parameters(policy_model_, "value_head.")),
      ngp_head_parameters_(collect_head_parameters(policy_model_, "next_goal_head.")),
      trunk_optimizer_(
          trunk_parameters_,
          torch::optim::AdamWOptions(config_.offline_optimization.trunk_learning_rate)
              .weight_decay(config_.offline_optimization.trunk_weight_decay)),
      policy_head_optimizer_(
          policy_head_parameters_,
          torch::optim::AdamWOptions(config_.behavior_cloning.learning_rate)
              .weight_decay(config_.behavior_cloning.weight_decay)),
      value_head_optimizer_(
          value_head_parameters_,
          torch::optim::AdamWOptions(config_.value_pretraining.learning_rate)
              .weight_decay(config_.value_pretraining.weight_decay)),
      ngp_head_optimizer_(
          ngp_head_parameters_,
          torch::optim::AdamWOptions(config_.next_goal_predictor.learning_rate)
              .weight_decay(config_.next_goal_predictor.weight_decay)),
      device_(config_.ppo.device) {
  validate_config();
  maybe_initialize_from_checkpoint();
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
  if (config_.next_goal_predictor.enabled && train_dataset_.next_goal_classes() != 3) {
    throw std::runtime_error("Offline manifest next_goal_classes must be 3 for the shared next-goal head.");
  }
  if (config_.value_pretraining.enabled && !train_dataset_.has_trajectory_end_flags()) {
    throw std::runtime_error(
        "Offline value pretraining requires terminated_path and truncated_path entries in every shard. "
        "Re-run scripts/preprocess_kaggle_2v2.py with the current repo version.");
  }
}

void OfflinePretrainer::maybe_initialize_from_checkpoint() {
  if (config_.next_goal_predictor.init_checkpoint.empty()) {
    return;
  }

  namespace fs = std::filesystem;
  const fs::path base(config_.next_goal_predictor.init_checkpoint);
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_init_checkpoint_compatibility(metadata, checkpoint_config, config_, "Next-goal init checkpoint");

  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string());
  policy_model_->load(archive);
  normalizer_.load(archive);

  auto maybe_load_optimizer = [&](const fs::path& path, torch::optim::AdamW& optimizer) {
    if (!fs::exists(path)) {
      return;
    }
    torch::serialize::InputArchive optimizer_archive;
    optimizer_archive.load_from(path.string());
    optimizer.load(optimizer_archive);
  };

  maybe_load_optimizer(base / "trunk_optimizer.pt", trunk_optimizer_);
  maybe_load_optimizer(base / "policy_head_optimizer.pt", policy_head_optimizer_);
  maybe_load_optimizer(base / "value_head_optimizer.pt", value_head_optimizer_);
  maybe_load_optimizer(base / "ngp_head_optimizer.pt", ngp_head_optimizer_);

  std::cout << "initialized_next_goal_from_checkpoint=" << base.string() << '\n';
}

void OfflinePretrainer::fit_normalizer() {
  PULSAR_TRACE_SCOPE_CAT("offline", "fit_normalizer");
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
    int epoch_index,
    SharedActorCritic target_model,
    const ObservationNormalizer& target_normalizer,
    const std::function<void(const OfflineEpochMetrics&, double, std::int64_t)>& progress_callback) {
  PULSAR_TRACE_SCOPE_CAT("offline", training ? "run_training_epoch" : "run_eval_epoch");
  OfflineEpochMetrics metrics{};
  const auto epoch_start = std::chrono::steady_clock::now();
  const bool should_report_progress =
      static_cast<bool>(progress_callback) && config_.wandb.log_interval_seconds > 0.0;
  const double progress_interval_seconds = std::max(1.0, config_.wandb.log_interval_seconds);
  double last_progress_report_seconds = 0.0;
  const bool train_policy = training && config_.behavior_cloning.enabled &&
                            epoch_index <= config_.behavior_cloning.epochs;
  const bool train_ngp = training && config_.next_goal_predictor.enabled &&
                         epoch_index <= config_.next_goal_predictor.epochs;
  const bool train_value = training && config_.value_pretraining.enabled &&
                           epoch_index <= config_.value_pretraining.epochs;
  const bool eval_policy = config_.behavior_cloning.enabled;
  const bool eval_ngp = config_.next_goal_predictor.enabled;
  const bool eval_value = config_.value_pretraining.enabled && static_cast<bool>(target_model);
  const bool shuffle = training ? config_.offline_dataset.shuffle : false;
  const std::uint64_t seed = config_.offline_dataset.seed + static_cast<std::uint64_t>(epoch_index);
  const torch::Tensor ngp_class_weights =
      torch::tensor(config_.next_goal_predictor.class_weights, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_);
  const std::int64_t sequence_length = std::max<std::int64_t>(1, config_.behavior_cloning.sequence_length);
  auto maybe_report_progress = [&]() {
    if (!should_report_progress || metrics.samples <= 0) {
      return;
    }
    const double elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    if (elapsed_seconds - last_progress_report_seconds < progress_interval_seconds) {
      return;
    }
    progress_callback(metrics, elapsed_seconds, dataset.sample_count());
    last_progress_report_seconds = elapsed_seconds;
  };

  dataset.for_each_packed_trajectory_batch(
      config_.offline_dataset.batch_size,
      shuffle,
      seed,
      [&](const OfflineTensorPackedBatch& batch) {
        PULSAR_TRACE_SCOPE_CAT("offline", "trajectory_batch");
        torch::Tensor obs;
        torch::Tensor weights;
        torch::Tensor normalized;
        torch::Tensor starts;
        torch::Tensor valid_mask;
        {
          PULSAR_TRACE_SCOPE_CAT("offline", "copy_and_normalize");
          obs = batch.obs.to(device_);
          weights = batch.weights.to(device_);
          normalized = normalizer_.normalize(obs).contiguous();
          starts = batch.episode_starts.to(device_).to(torch::kFloat32).contiguous();
          valid_mask = batch.valid_mask.to(device_).contiguous();
        }
        const std::int64_t max_time = normalized.size(0);
        const std::int64_t packed_count = normalized.size(1);
        if (max_time <= 0 || packed_count <= 0) {
          return;
        }

        torch::Tensor value_targets =
            torch::zeros({max_time, packed_count}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        torch::Tensor value_mask =
            torch::zeros({max_time, packed_count}, torch::TensorOptions().dtype(torch::kBool).device(device_));
        if (eval_value && max_time > 1) {
          PULSAR_TRACE_SCOPE_CAT("offline", "target_value_build");
          const torch::Tensor terminated = batch.terminated.to(device_).to(torch::kFloat32).contiguous();
          const torch::Tensor truncated = batch.truncated.to(device_).to(torch::kFloat32).contiguous();
          ContinuumState target_state = target_model->initial_state(packed_count, device_);
          torch::NoGradGuard no_grad;
          const torch::Tensor target_normalized = target_normalizer.normalize(obs).contiguous();
          SequenceOutput target_output = target_model->forward_sequence(target_normalized, target_state, starts);
          const torch::Tensor target_scores = ngp_scalar(target_output.next_goal_logits);
          const torch::Tensor transition_mask =
              valid_mask.narrow(0, 0, max_time - 1).logical_and(valid_mask.narrow(0, 1, max_time - 1));
          const torch::Tensor rewards =
              config_.reward.ngp_scale *
              (target_scores.narrow(0, 1, max_time - 1) - target_scores.narrow(0, 0, max_time - 1));

          std::vector<std::int64_t> last_step_values;
          last_step_values.reserve(batch.lengths.size());
          for (const std::int64_t length : batch.lengths) {
            last_step_values.push_back(std::max<std::int64_t>(0, length - 1));
          }

          const torch::Tensor last_indices =
              torch::tensor(last_step_values, torch::TensorOptions().dtype(torch::kLong).device(device_));
          torch::Tensor bootstrap =
              torch::zeros({packed_count}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
          if (config_.value_pretraining.bootstrap_truncated) {
            const torch::Tensor last_truncated =
                truncated.transpose(0, 1).gather(1, last_indices.unsqueeze(1)).squeeze(1);
            const torch::Tensor last_terminated =
                terminated.transpose(0, 1).gather(1, last_indices.unsqueeze(1)).squeeze(1);
            const torch::Tensor value_logits_bta = target_output.value_logits.permute({1, 0, 2});
            const std::int64_t atom_count = value_logits_bta.size(2);
            const torch::Tensor gathered_last_logits =
                value_logits_bta
                    .gather(
                        1,
                        last_indices.view({packed_count, 1, 1}).expand({packed_count, 1, atom_count}))
                    .squeeze(1);
            const torch::Tensor bootstrap_mask =
                last_truncated.gt(0.5F).logical_and(last_terminated.le(0.5F));
            const torch::Tensor bootstrap_values = target_model->expected_value(gathered_last_logits);
            bootstrap = torch::where(
                bootstrap_mask,
                bootstrap_values,
                torch::zeros_like(bootstrap_values));
          }

          torch::Tensor running = bootstrap;
          for (std::int64_t t = max_time - 2; t >= 0; --t) {
            const torch::Tensor active = transition_mask.select(0, t);
            running = torch::where(active, rewards.select(0, t) + config_.ppo.gamma * running, running);
            value_targets.select(0, t).copy_(
                torch::where(active, running, torch::zeros_like(running)));
            value_mask.select(0, t).copy_(active);
          }
        }

        ContinuumState state = policy_model_->initial_state(packed_count, device_);
        const torch::Tensor actions = batch.actions.defined() ? batch.actions.to(device_) : torch::Tensor{};
        const torch::Tensor action_probs =
            batch.action_probs.defined() ? batch.action_probs.to(device_) : torch::Tensor{};
        const torch::Tensor next_goal = batch.next_goal.defined() ? batch.next_goal.to(device_) : torch::Tensor{};

        for (std::int64_t offset = 0; offset < max_time; offset += sequence_length) {
          PULSAR_TRACE_SCOPE_CAT("offline", "chunk");
          const std::int64_t chunk_length = std::min<std::int64_t>(sequence_length, max_time - offset);
          const std::int64_t valid_rows = count_valid_rows_in_window(batch.lengths, offset, chunk_length);
          if (valid_rows <= 0) {
            continue;
          }
          const std::int64_t active_value_rows = count_value_rows_in_window(batch.lengths, offset, chunk_length);
          const torch::Tensor chunk_obs = normalized.narrow(0, offset, chunk_length);
          const torch::Tensor chunk_starts = starts.narrow(0, offset, chunk_length);
          const torch::Tensor chunk_weights = weights.narrow(0, offset, chunk_length);
          const torch::Tensor chunk_valid = valid_mask.narrow(0, offset, chunk_length);
          const bool has_policy_targets = actions.defined() || action_probs.defined();
          const bool has_ngp_targets = next_goal.defined();
          const torch::Tensor chunk_value_mask = value_mask.narrow(0, offset, chunk_length);
          const bool has_value_targets = active_value_rows > 0;

          auto compute_losses = [&](const SequenceOutput& output, torch::Tensor* total_loss) {
            if (eval_policy && has_policy_targets) {
              const torch::Tensor flat_valid = chunk_valid.reshape({-1});
              const torch::Tensor flat_weights = chunk_weights.reshape({-1}).index({flat_valid});
              const torch::Tensor flat_logits =
                  output.policy_logits.reshape({-1, output.policy_logits.size(2)}).index({flat_valid});
              if (flat_logits.size(0) > 0) {
                torch::Tensor loss;
                if (action_probs.defined()) {
                  const torch::Tensor flat_action_probs =
                      action_probs.narrow(0, offset, chunk_length)
                          .reshape({-1, action_probs.size(2)})
                          .index({flat_valid});
                  loss = weighted_soft_cross_entropy(flat_logits, flat_action_probs, flat_weights);
                  metrics.policy_accuracy +=
                      weighted_accuracy_from_probs(flat_logits, flat_action_probs, flat_weights) *
                      static_cast<double>(flat_logits.size(0));
                } else {
                  const torch::Tensor flat_actions =
                      actions.narrow(0, offset, chunk_length).reshape({-1}).index({flat_valid});
                  loss = weighted_cross_entropy(
                      flat_logits,
                      flat_actions,
                      flat_weights,
                      config_.behavior_cloning.label_smoothing);
                  metrics.policy_accuracy +=
                      weighted_accuracy(flat_logits, flat_actions, flat_weights) *
                      static_cast<double>(flat_logits.size(0));
                }
                metrics.policy_loss += loss.item<double>() * static_cast<double>(flat_logits.size(0));
                metrics.policy_samples += flat_logits.size(0);
                if (train_policy && total_loss != nullptr) {
                  *total_loss = *total_loss + loss;
                }
              }
            }

            if (eval_ngp && has_ngp_targets) {
              const torch::Tensor flat_valid = chunk_valid.reshape({-1});
              const torch::Tensor flat_weights = chunk_weights.reshape({-1}).index({flat_valid});
              const torch::Tensor flat_labels =
                  next_goal.narrow(0, offset, chunk_length).reshape({-1}).index({flat_valid});
              const torch::Tensor flat_logits =
                  output.next_goal_logits.reshape({-1, output.next_goal_logits.size(2)}).index({flat_valid});
              if (flat_logits.size(0) > 0) {
                const torch::Tensor loss = weighted_cross_entropy(
                    flat_logits,
                    flat_labels,
                    flat_weights,
                    config_.next_goal_predictor.label_smoothing,
                    ngp_class_weights);
                metrics.ngp_loss += loss.item<double>() * static_cast<double>(flat_logits.size(0));
                metrics.ngp_accuracy +=
                    weighted_accuracy(flat_logits, flat_labels, flat_weights) * static_cast<double>(flat_logits.size(0));
                metrics.ngp_samples += flat_logits.size(0);
                if (train_ngp && total_loss != nullptr) {
                  *total_loss = *total_loss + loss;
                }
              }
            }

            if (eval_value && has_value_targets) {
              const torch::Tensor active = chunk_value_mask.reshape({-1});
              const torch::Tensor active_logits =
                  output.value_logits.reshape({-1, output.value_logits.size(2)}).index({active});
              if (active_logits.size(0) > 0) {
                const torch::Tensor active_returns =
                    value_targets.narrow(0, offset, chunk_length).reshape({-1}).index({active});
                const torch::Tensor active_weights = chunk_weights.reshape({-1}).index({active});
                const torch::Tensor target_dist = categorical_projection(config_, active_returns);
                const torch::Tensor per_sample =
                    -(target_dist * torch::log_softmax(active_logits, -1)).sum(-1);
                const torch::Tensor normalized_weights = active_weights / active_weights.mean().clamp_min(1.0e-6);
                const torch::Tensor loss = (per_sample * normalized_weights).mean();
                const auto value_samples = active_returns.size(0);
                metrics.value_loss += loss.item<double>() * static_cast<double>(value_samples);
                metrics.value_samples += value_samples;
                if (train_value && total_loss != nullptr) {
                  *total_loss = *total_loss + config_.value_pretraining.loss_coef * loss;
                }
              }
            }
          };

          if (!training) {
            torch::NoGradGuard no_grad;
            SequenceOutput output;
            {
              PULSAR_TRACE_SCOPE_CAT("offline", "sequence_forward");
              output = policy_model_->forward_sequence(chunk_obs, state, chunk_starts);
            }
            compute_losses(output, nullptr);
            state = detach_state(std::move(output.final_state));
            metrics.samples += valid_rows;
            maybe_report_progress();
            continue;
          }

          const bool chunk_has_training_loss =
              (train_policy && has_policy_targets) || (train_ngp && has_ngp_targets) || (train_value && has_value_targets);

          if (chunk_has_training_loss) {
            policy_model_->zero_grad();
            trunk_optimizer_.zero_grad();
            policy_head_optimizer_.zero_grad();
            value_head_optimizer_.zero_grad();
            ngp_head_optimizer_.zero_grad();

            SequenceOutput output;
            {
              PULSAR_TRACE_SCOPE_CAT("offline", "sequence_forward");
              output = policy_model_->forward_sequence(chunk_obs, state, chunk_starts);
            }
            torch::Tensor total_loss =
                torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            {
              PULSAR_TRACE_SCOPE_CAT("offline", "loss_math");
              compute_losses(output, &total_loss);
            }

            if (total_loss.requires_grad()) {
              {
                PULSAR_TRACE_SCOPE_CAT("offline", "backward");
                total_loss.backward();
                if (!trunk_parameters_.empty()) {
                  torch::nn::utils::clip_grad_norm_(
                      trunk_parameters_,
                      config_.offline_optimization.trunk_max_grad_norm);
                }
                if (train_policy && !policy_head_parameters_.empty()) {
                  torch::nn::utils::clip_grad_norm_(
                      policy_head_parameters_,
                      config_.behavior_cloning.max_grad_norm);
                }
                if (train_value && !value_head_parameters_.empty()) {
                  torch::nn::utils::clip_grad_norm_(
                      value_head_parameters_,
                      config_.value_pretraining.max_grad_norm);
                }
                if (train_ngp && !ngp_head_parameters_.empty()) {
                  torch::nn::utils::clip_grad_norm_(
                      ngp_head_parameters_,
                      config_.next_goal_predictor.max_grad_norm);
                }
              }
              {
                PULSAR_TRACE_SCOPE_CAT("offline", "optimizer_step");
                trunk_optimizer_.step();
                if (train_policy && has_policy_targets) {
                  policy_head_optimizer_.step();
                }
                if (train_value && has_value_targets) {
                  value_head_optimizer_.step();
                }
                if (train_ngp && has_ngp_targets) {
                  ngp_head_optimizer_.step();
                }
              }
            }
            state = detach_state(std::move(output.final_state));
          } else {
            torch::NoGradGuard no_grad;
            SequenceOutput output;
            {
              PULSAR_TRACE_SCOPE_CAT("offline", "sequence_forward");
              output = policy_model_->forward_sequence(chunk_obs, state, chunk_starts);
            }
            compute_losses(output, nullptr);
            state = detach_state(std::move(output.final_state));
          }

          metrics.samples += valid_rows;
          maybe_report_progress();
        }
      });
  return averaged_epoch_metrics(metrics);
}

OfflineEpochMetrics OfflinePretrainer::run_training_epoch(
    int epoch_index,
    SharedActorCritic target_model,
    const ObservationNormalizer& target_normalizer,
    const std::function<void(const OfflineEpochMetrics&, double, std::int64_t)>& progress_callback) {
  return run_epoch(train_dataset_, true, epoch_index, target_model, target_normalizer, progress_callback);
}

OfflineEpochMetrics OfflinePretrainer::evaluate(
    SharedActorCritic target_model,
    const ObservationNormalizer& target_normalizer,
    const std::function<void(const OfflineEpochMetrics&, double, std::int64_t)>& progress_callback) {
  return run_epoch(val_dataset_, false, 0, target_model, target_normalizer, progress_callback);
}

void OfflinePretrainer::save_checkpoint(const std::string& output_dir, int epoch_index) const {
  PULSAR_TRACE_SCOPE_CAT("checkpoint", "save_offline_checkpoint");
  namespace fs = std::filesystem;
  const fs::path base = fs::path(output_dir);
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

  torch::serialize::OutputArchive trunk_optimizer_archive;
  trunk_optimizer_.save(trunk_optimizer_archive);
  trunk_optimizer_archive.save_to((base / "trunk_optimizer.pt").string());

  torch::serialize::OutputArchive policy_head_optimizer_archive;
  policy_head_optimizer_.save(policy_head_optimizer_archive);
  policy_head_optimizer_archive.save_to((base / "policy_head_optimizer.pt").string());

  torch::serialize::OutputArchive value_head_optimizer_archive;
  value_head_optimizer_.save(value_head_optimizer_archive);
  value_head_optimizer_archive.save_to((base / "value_head_optimizer.pt").string());

  torch::serialize::OutputArchive ngp_head_optimizer_archive;
  ngp_head_optimizer_.save(ngp_head_optimizer_archive);
  ngp_head_optimizer_archive.save_to((base / "ngp_head_optimizer.pt").string());
}

void OfflinePretrainer::train(const std::string& output_dir, const std::string& config_path) {
  PULSAR_TRACE_SCOPE_CAT("offline", "train");
  if (config_.next_goal_predictor.init_checkpoint.empty() || !config_.next_goal_predictor.reuse_normalizer) {
    fit_normalizer();
  }
  WandbLogger wandb(config_.wandb, output_dir, config_path, "offline_pretrain");
  auto make_live_progress_callback = [&](const char* phase, int epoch_index) {
    const std::string phase_name(phase);
    return [&, phase_name, epoch_index](const OfflineEpochMetrics& metrics, double elapsed_seconds, std::int64_t total_samples) {
      if (!wandb.enabled()) {
        return;
      }
      wandb.log(make_live_offline_metrics_payload(
          phase_name.c_str(),
          epoch_index,
          metrics,
          elapsed_seconds,
          total_samples));
    };
  };

  const int max_epochs = std::max({config_.behavior_cloning.epochs, config_.next_goal_predictor.epochs, config_.value_pretraining.epochs});
  if (max_epochs <= 0) {
    SharedActorCritic eval_target = nullptr;
    ObservationNormalizer eval_normalizer = normalizer_.clone();
    if (config_.value_pretraining.enabled) {
      eval_target = clone_shared_model(policy_model_, device_);
      eval_target->eval();
      eval_normalizer.to(device_);
    }
    policy_model_->eval();
    const OfflineEpochMetrics val_metrics = evaluate(
        eval_target,
        eval_normalizer,
        wandb.enabled() ? make_live_progress_callback("val", 0)
                        : std::function<void(const OfflineEpochMetrics&, double, std::int64_t)>{});
    const OfflineEpochMetrics train_metrics{};
    append_offline_metrics_line(output_dir, 0, train_metrics, val_metrics);
    if (wandb.enabled()) {
      wandb.log(nlohmann::json{
          {"epoch", 0},
          {"train_policy_loss", train_metrics.policy_loss},
          {"train_ngp_loss", train_metrics.ngp_loss},
          {"train_value_loss", train_metrics.value_loss},
          {"train_policy_accuracy", train_metrics.policy_accuracy},
          {"train_ngp_accuracy", train_metrics.ngp_accuracy},
          {"train_samples", train_metrics.samples},
          {"train_policy_samples", train_metrics.policy_samples},
          {"train_ngp_samples", train_metrics.ngp_samples},
          {"train_value_samples", train_metrics.value_samples},
          {"val_policy_loss", val_metrics.policy_loss},
          {"val_ngp_loss", val_metrics.ngp_loss},
          {"val_value_loss", val_metrics.value_loss},
          {"val_policy_accuracy", val_metrics.policy_accuracy},
          {"val_ngp_accuracy", val_metrics.ngp_accuracy},
          {"val_samples", val_metrics.samples},
          {"val_policy_samples", val_metrics.policy_samples},
          {"val_ngp_samples", val_metrics.ngp_samples},
          {"val_value_samples", val_metrics.value_samples},
          {"epoch_seconds", 0.0},
      });
    }
    wandb.finish();
    return;
  }

  SharedActorCritic training_target = nullptr;
  ObservationNormalizer training_target_normalizer = normalizer_.clone();
  int target_epoch = 0;
  for (int epoch = 1; epoch <= max_epochs; ++epoch) {
    PULSAR_TRACE_SCOPE_CAT("offline", "epoch");
    if (config_.value_pretraining.enabled &&
        (!training_target || epoch == 1 ||
         epoch - target_epoch >= std::max(1, config_.value_pretraining.target_sync_interval_epochs))) {
      training_target = clone_shared_model(policy_model_, device_);
      training_target->eval();
      training_target_normalizer = normalizer_.clone();
      training_target_normalizer.to(device_);
      target_epoch = epoch;
    }

    const auto start = std::chrono::steady_clock::now();
    policy_model_->train();
    const OfflineEpochMetrics train_metrics = run_training_epoch(
        epoch,
        training_target,
        training_target_normalizer,
        wandb.enabled() ? make_live_progress_callback("train", epoch)
                        : std::function<void(const OfflineEpochMetrics&, double, std::int64_t)>{});

    SharedActorCritic eval_target = nullptr;
    ObservationNormalizer eval_normalizer = normalizer_.clone();
    if (config_.value_pretraining.enabled) {
      eval_target = clone_shared_model(policy_model_, device_);
      eval_target->eval();
      eval_normalizer.to(device_);
    }
    policy_model_->eval();
    const OfflineEpochMetrics val_metrics = evaluate(
        eval_target,
        eval_normalizer,
        wandb.enabled() ? make_live_progress_callback("val", epoch)
                        : std::function<void(const OfflineEpochMetrics&, double, std::int64_t)>{});
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "epoch=" << epoch
              << " seconds=" << seconds
              << " train_policy_loss=" << train_metrics.policy_loss
              << " train_ngp_loss=" << train_metrics.ngp_loss
              << " train_value_loss=" << train_metrics.value_loss
              << " train_policy_acc=" << train_metrics.policy_accuracy
              << " train_ngp_acc=" << train_metrics.ngp_accuracy
              << " val_policy_loss=" << val_metrics.policy_loss
              << " val_ngp_loss=" << val_metrics.ngp_loss
              << " val_value_loss=" << val_metrics.value_loss
              << " val_policy_acc=" << val_metrics.policy_accuracy
              << " val_ngp_acc=" << val_metrics.ngp_accuracy
              << '\n';

    append_offline_metrics_line(output_dir, epoch, train_metrics, val_metrics);
    if (wandb.enabled()) {
      wandb.log(nlohmann::json{
          {"epoch", epoch},
          {"train_policy_loss", train_metrics.policy_loss},
          {"train_ngp_loss", train_metrics.ngp_loss},
          {"train_value_loss", train_metrics.value_loss},
          {"train_policy_accuracy", train_metrics.policy_accuracy},
          {"train_ngp_accuracy", train_metrics.ngp_accuracy},
          {"train_samples", train_metrics.samples},
          {"train_policy_samples", train_metrics.policy_samples},
          {"train_ngp_samples", train_metrics.ngp_samples},
          {"train_value_samples", train_metrics.value_samples},
          {"val_policy_loss", val_metrics.policy_loss},
          {"val_ngp_loss", val_metrics.ngp_loss},
          {"val_value_loss", val_metrics.value_loss},
          {"val_policy_accuracy", val_metrics.policy_accuracy},
          {"val_ngp_accuracy", val_metrics.ngp_accuracy},
          {"val_samples", val_metrics.samples},
          {"val_policy_samples", val_metrics.policy_samples},
          {"val_ngp_samples", val_metrics.ngp_samples},
          {"val_value_samples", val_metrics.value_samples},
          {"epoch_seconds", seconds},
      });
    }
    if (config_.behavior_cloning.enabled || config_.next_goal_predictor.enabled || config_.value_pretraining.enabled) {
      save_checkpoint(output_dir, epoch);
    }
  }
  wandb.finish();
}

OfflineBenchmarkMetrics OfflinePretrainer::benchmark(int warmup_epochs, int measured_epochs) {
  PULSAR_TRACE_SCOPE_CAT("offline", "benchmark");
  const int warmup = std::max(0, warmup_epochs);
  const int measured = std::max(1, measured_epochs);

  OfflineBenchmarkMetrics setup_metrics{};
  if (config_.next_goal_predictor.init_checkpoint.empty() || !config_.next_goal_predictor.reuse_normalizer) {
    const auto fit_start = std::chrono::steady_clock::now();
    fit_normalizer();
    setup_metrics.fit_normalizer_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - fit_start).count();
  }

  auto run_benchmark_epoch = [&](int epoch) {
    OfflineBenchmarkMetrics metrics{};
    SharedActorCritic training_target = nullptr;
    ObservationNormalizer training_target_normalizer = normalizer_.clone();
    if (config_.value_pretraining.enabled) {
      training_target = clone_shared_model(policy_model_, device_);
      training_target->eval();
      training_target_normalizer.to(device_);
    }

    const auto train_start = std::chrono::steady_clock::now();
    policy_model_->train();
    const OfflineEpochMetrics train_metrics = run_training_epoch(epoch, training_target, training_target_normalizer);
    metrics.train_epoch_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - train_start).count();
    metrics.train_samples = train_metrics.samples;

    SharedActorCritic eval_target = nullptr;
    ObservationNormalizer eval_normalizer = normalizer_.clone();
    if (config_.value_pretraining.enabled) {
      eval_target = clone_shared_model(policy_model_, device_);
      eval_target->eval();
      eval_normalizer.to(device_);
    }

    const auto eval_start = std::chrono::steady_clock::now();
    policy_model_->eval();
    const OfflineEpochMetrics eval_metrics = evaluate(eval_target, eval_normalizer);
    metrics.eval_epoch_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();
    metrics.eval_samples = eval_metrics.samples;
    metrics.overall_seconds = metrics.train_epoch_seconds + metrics.eval_epoch_seconds;
    return metrics;
  };

  for (int epoch = 1; epoch <= warmup; ++epoch) {
    PULSAR_TRACE_SCOPE_CAT("offline", "benchmark_warmup");
    (void)run_benchmark_epoch(epoch);
  }

  OfflineBenchmarkMetrics aggregate = setup_metrics;
  for (int index = 0; index < measured; ++index) {
    PULSAR_TRACE_SCOPE_CAT("offline", "benchmark_measure");
    const OfflineBenchmarkMetrics epoch_metrics = run_benchmark_epoch(warmup + index + 1);
    accumulate_offline_benchmark_metrics(&aggregate, epoch_metrics);
  }
  average_offline_benchmark_metrics(&aggregate, static_cast<double>(measured));
  return aggregate;
}

}  // namespace pulsar

#endif
