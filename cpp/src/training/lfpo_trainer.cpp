#include "pulsar/training/lfpo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <ATen/Context.h>

#include "pulsar/tracing/tracing.hpp"
#include "pulsar/training/lfpo_math.hpp"

namespace pulsar {
namespace {

torch::Device resolve_runtime_device(const std::string& device_name) {
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

RolloutStorage make_rollout_storage(
    const ExperimentConfig& config,
    int num_agents,
    int action_dim) {
  return RolloutStorage(
      config.lfpo.rollout_length,
      num_agents,
      config.model.observation_dim,
      action_dim,
      config.lfpo.candidate_count,
      torch::Device(torch::kCPU));
}

void freeze_parameters(const FutureEvaluator& evaluator) {
  for (auto& parameter : evaluator->parameters()) {
    parameter.set_requires_grad(false);
  }
}

void configure_cuda_runtime_for_h100(const torch::Device& device) {
  if (!device.is_cuda()) {
    return;
  }
  at::globalContext().setAllowTF32CuBLAS(true);
  at::globalContext().setAllowTF32CuDNN(true);
}

void append_metrics_line(
    const std::filesystem::path& checkpoint_dir,
    int update_index,
    std::int64_t global_step,
    const TrainerMetrics& metrics) {
  nlohmann::json line = {
      {"update", update_index},
      {"global_step", global_step},
      {"collection_agent_steps_per_second", metrics.collection_agent_steps_per_second},
      {"update_agent_steps_per_second", metrics.update_agent_steps_per_second},
      {"overall_agent_steps_per_second", metrics.overall_agent_steps_per_second},
      {"update_seconds", metrics.update_seconds},
      {"policy_loss", metrics.policy_loss},
      {"latent_loss", metrics.latent_loss},
      {"evaluator_loss", metrics.evaluator_loss},
      {"entropy", metrics.entropy},
      {"obs_build_seconds", metrics.obs_build_seconds},
      {"mask_build_seconds", metrics.mask_build_seconds},
      {"policy_forward_seconds", metrics.policy_forward_seconds},
      {"action_decode_seconds", metrics.action_decode_seconds},
      {"env_step_seconds", metrics.env_step_seconds},
      {"done_reset_seconds", metrics.done_reset_seconds},
      {"lfpo_forward_backward_seconds", metrics.lfpo_forward_backward_seconds},
      {"optimizer_step_seconds", metrics.optimizer_step_seconds},
      {"self_play_eval_seconds", metrics.self_play_eval_seconds},
      {"online_outcome_samples", metrics.online_outcome_samples},
      {"online_outcome_trajectories", metrics.online_outcome_trajectories},
      {"evaluator_target_update_index", metrics.evaluator_target_update_index},
  };
  for (const auto& [mode, rating] : metrics.elo_ratings) {
    line["elo_" + mode] = rating;
  }
  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

struct RolloutFutureTargets {
  torch::Tensor embeddings;
  torch::Tensor horizon_mask;
  torch::Tensor known_mask;
};

RolloutFutureTargets build_rollout_targets(
    const RolloutStorage& rollout,
    const ExperimentConfig& config,
    const ObservationNormalizer& evaluator_normalizer,
    FutureEvaluator target_evaluator,
    const torch::Device& device) {
  const int max_h = target_evaluator->max_horizon();
  const int horizon_count = static_cast<int>(config.future_evaluator.horizons.size());
  const int time = rollout.rollout_length();
  const int agents = rollout.num_agents();
  const int obs_dim = config.model.observation_dim;
  torch::Tensor embeddings = torch::zeros(
      {time, agents, horizon_count, config.future_evaluator.latent_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor horizon_mask = torch::zeros(
      {time, agents, horizon_count},
      torch::TensorOptions().dtype(torch::kBool).device(device));
  torch::Tensor known_mask = torch::zeros(
      {time, agents},
      torch::TensorOptions().dtype(torch::kBool).device(device));

  std::vector<torch::Tensor> windows;
  std::vector<std::int64_t> row_t;
  std::vector<std::int64_t> row_a;
  std::vector<std::vector<bool>> row_horizon_masks;
  const torch::Tensor raw = rollout.raw_obs.to(torch::kCPU);
  const torch::Tensor dones = rollout.dones.to(torch::kCPU);
  const torch::Tensor labels = rollout.terminal_outcomes.to(torch::kCPU);

  for (int agent = 0; agent < agents; ++agent) {
    for (int t = 0; t < time; ++t) {
      int done_step = -1;
      std::int64_t outcome = 2;
      for (int future_t = t; future_t < time; ++future_t) {
        if (dones[future_t][agent].item<float>() > 0.5F) {
          done_step = future_t;
          outcome = labels[future_t][agent].item<std::int64_t>();
          break;
        }
      }
      if (done_step < 0) {
        continue;
      }
      torch::Tensor window = torch::zeros({max_h + 1, obs_dim}, raw.options());
      for (int dt = 0; dt <= max_h; ++dt) {
        const int src_t = std::min(t + dt, done_step);
        window[dt].copy_(raw[src_t][agent]);
      }
      std::vector<bool> masks;
      masks.reserve(static_cast<std::size_t>(horizon_count));
      for (const int horizon : config.future_evaluator.horizons) {
        const bool terminal_known = outcome == 0 || outcome == 1;
        masks.push_back(terminal_known || (t + horizon <= done_step));
      }
      windows.push_back(window);
      row_t.push_back(t);
      row_a.push_back(agent);
      row_horizon_masks.push_back(std::move(masks));
    }
  }

  if (windows.empty()) {
    return {embeddings, horizon_mask, known_mask};
  }

  torch::NoGradGuard no_grad;
  const torch::Tensor stacked = torch::stack(windows, 0).to(device);
  const torch::Tensor normalized = evaluator_normalizer
                                       .normalize(stacked.reshape({-1, obs_dim}))
                                       .reshape_as(stacked);
  const FutureEvaluationOutput output = target_evaluator->forward_windows(normalized);
  for (std::size_t row = 0; row < row_t.size(); ++row) {
    embeddings[row_t[row]][row_a[row]].copy_(output.embeddings[static_cast<long>(row)]);
    known_mask[row_t[row]][row_a[row]] = true;
    for (int h = 0; h < horizon_count; ++h) {
      if (row_horizon_masks[row][static_cast<std::size_t>(h)]) {
        horizon_mask[row_t[row]][row_a[row]][h] = true;
      }
    }
  }
  return {embeddings, horizon_mask, known_mask};
}

}  // namespace

LFPOTrainer::LFPOTrainer(
    ExperimentConfig config,
    std::unique_ptr<BatchedRocketSimCollector> collector,
    std::unique_ptr<SelfPlayManager> self_play_manager,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : config_(std::move(config)),
      collector_(std::move(collector)),
      self_play_manager_(std::move(self_play_manager)),
      action_table_(config_.action_table),
      actor_(LatentFutureActor(config_.model)),
      evaluator_(FutureEvaluator(config_.future_evaluator, config_.model.observation_dim)),
      target_evaluator_(FutureEvaluator(config_.future_evaluator, config_.model.observation_dim)),
      actor_normalizer_(config_.model.observation_dim),
      evaluator_normalizer_(config_.model.observation_dim),
      actor_optimizer_(actor_->parameters(), torch::optim::AdamOptions(config_.lfpo.learning_rate)),
      evaluator_optimizer_(
          evaluator_->parameters(),
          torch::optim::AdamWOptions(config_.future_evaluator.learning_rate)
              .weight_decay(config_.future_evaluator.weight_decay)),
      rollout_(make_rollout_storage(
          config_,
          static_cast<int>(collector_->total_agents()),
          collector_->action_dim())),
      device_(resolve_runtime_device(config_.lfpo.device)),
      run_output_root_(std::move(run_output_root)),
      log_initialization_(log_initialization) {
  if (!collector_) {
    throw std::invalid_argument("LFPOTrainer requires a collector.");
  }
  total_agents_ = collector_->total_agents();
  collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  opponent_collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  collection_trajectory_ids_ =
      torch::arange(static_cast<long>(total_agents_), torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
  next_trajectory_id_ = static_cast<std::int64_t>(total_agents_);
  configure_cuda_runtime_for_h100(device_);
  use_pinned_host_buffers_ = device_.is_cuda();
  actor_->to(device_);
  evaluator_->to(device_);
  target_evaluator_->to(device_);
  actor_normalizer_.to(device_);
  evaluator_normalizer_.to(device_);
  outcome_buffer_ = std::make_unique<OnlineOutcomeReplayBuffer>(
      config_.model.observation_dim,
      collector_->num_envs(),
      collector_->num_envs() == 0 ? 0 : total_agents_ / collector_->num_envs(),
      config_.lfpo.online_window_capacity);
  maybe_initialize_from_checkpoint();
  target_evaluator_ = clone_future_evaluator(evaluator_, device_);
  target_evaluator_->eval();
  freeze_parameters(target_evaluator_);

  if (self_play_manager_ && self_play_manager_->enabled()) {
    collector_->set_self_play_assignment_fn(
        [this](std::size_t env_idx, std::uint64_t seed) {
          return self_play_manager_->sample_assignment(env_idx, seed);
        });
  }
}

torch::Tensor LFPOTrainer::sample_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs) const {
  return sample_masked_actions(logits, action_masks, deterministic, log_probs);
}

torch::Tensor LFPOTrainer::sample_candidate_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    torch::Tensor* log_probs) const {
  std::vector<torch::Tensor> actions;
  std::vector<torch::Tensor> logs;
  actions.reserve(static_cast<std::size_t>(config_.lfpo.candidate_count));
  logs.reserve(static_cast<std::size_t>(config_.lfpo.candidate_count));
  for (int index = 0; index < config_.lfpo.candidate_count; ++index) {
    torch::Tensor current_log_probs;
    actions.push_back(sample_actions(logits, action_masks, false, &current_log_probs));
    logs.push_back(current_log_probs);
  }
  if (log_probs != nullptr) {
    *log_probs = torch::stack(logs, -1);
  }
  return torch::stack(actions, -1);
}

void LFPOTrainer::maybe_initialize_from_checkpoint() {
  if (config_.lfpo.init_checkpoint.empty()) {
    return;
  }
  const std::filesystem::path base(config_.lfpo.init_checkpoint);
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_inference_checkpoint_metadata(metadata, checkpoint_config);
  torch::serialize::InputArchive actor_archive;
  actor_archive.load_from((base / "model.pt").string(), device_);
  actor_->load(actor_archive);
  actor_normalizer_.load(actor_archive);
  actor_->to(device_);
  actor_normalizer_.to(device_);
  if (std::filesystem::exists(base / "actor_optimizer.pt")) {
    torch::serialize::InputArchive optimizer_archive;
    optimizer_archive.load_from((base / "actor_optimizer.pt").string(), device_);
    actor_optimizer_.load(optimizer_archive);
    resumed_global_step_ = metadata.global_step;
    resumed_update_index_ = metadata.update_index;
  }
  load_future_evaluator_checkpoint(base / "future_evaluator");
  if (log_initialization_) {
    std::cout << "initialized_lfpo_from_checkpoint=" << base.string() << '\n';
  }
}

void LFPOTrainer::load_future_evaluator_checkpoint(const std::filesystem::path& base) {
  if (!std::filesystem::exists(base / "model.pt")) {
    throw std::runtime_error("LFPO checkpoint is missing future_evaluator/model.pt.");
  }
  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string(), device_);
  evaluator_->load(archive);
  evaluator_normalizer_.load(archive);
  evaluator_->to(device_);
  evaluator_normalizer_.to(device_);
  if (std::filesystem::exists(base / "online_model.pt")) {
    torch::serialize::InputArchive online_archive;
    online_archive.load_from((base / "online_model.pt").string(), device_);
    evaluator_->load(online_archive);
  }
}

void LFPOTrainer::update_evaluator_from_self_play(int update_index, TrainerMetrics* metrics) {
  if (config_.lfpo.evaluator_update_interval <= 0 ||
      update_index % config_.lfpo.evaluator_update_interval != 0 ||
      !outcome_buffer_) {
    return;
  }
  const std::vector<OutcomeTrajectory> trajectories = outcome_buffer_->trajectories();
  if (trajectories.empty()) {
    return;
  }
  evaluator_->train();
  const torch::Tensor class_weights =
      torch::tensor(config_.future_evaluator.class_weights, torch::TensorOptions().dtype(torch::kFloat32)).to(device_);
  double total_loss = 0.0;
  std::int64_t total_samples = 0;
  for (const auto& trajectory : trajectories) {
    if (!trajectory.obs_cpu.defined() || trajectory.obs_cpu.size(0) <= 0) {
      continue;
    }
    const int max_h = evaluator_->max_horizon();
    const auto steps = trajectory.obs_cpu.size(0);
    std::vector<torch::Tensor> windows;
    std::vector<std::vector<bool>> masks;
    for (std::int64_t t = 0; t < steps; ++t) {
      torch::Tensor window = torch::zeros(
          {max_h + 1, config_.model.observation_dim},
          torch::TensorOptions().dtype(torch::kFloat32));
      for (int dt = 0; dt <= max_h; ++dt) {
        const std::int64_t src_t = std::min<std::int64_t>(t + dt, steps - 1);
        window[dt].copy_(trajectory.obs_cpu[src_t]);
      }
      std::vector<bool> horizon_masks;
      for (const int horizon : config_.future_evaluator.horizons) {
        horizon_masks.push_back(trajectory.outcome == 0 || trajectory.outcome == 1 || t + horizon < steps);
      }
      windows.push_back(window);
      masks.push_back(std::move(horizon_masks));
    }
    if (windows.empty()) {
      continue;
    }
    const torch::Tensor stacked = torch::stack(windows, 0).to(device_);
    const torch::Tensor normalized = evaluator_normalizer_
                                         .normalize(stacked.reshape({-1, config_.model.observation_dim}))
                                         .reshape_as(stacked);
    const FutureEvaluationOutput output = evaluator_->forward_windows(normalized);
    const torch::Tensor labels =
        torch::full({stacked.size(0)}, trajectory.outcome, torch::TensorOptions().dtype(torch::kLong).device(device_));
    torch::Tensor horizon_mask =
        torch::zeros({stacked.size(0), static_cast<long>(config_.future_evaluator.horizons.size())},
                     torch::TensorOptions().dtype(torch::kBool).device(device_));
    for (std::int64_t row = 0; row < horizon_mask.size(0); ++row) {
      for (std::int64_t h = 0; h < horizon_mask.size(1); ++h) {
        if (masks[static_cast<std::size_t>(row)][static_cast<std::size_t>(h)]) {
          horizon_mask[row][h] = true;
        }
      }
    }
    const torch::Tensor flat_logits = output.outcome_logits.reshape({-1, config_.future_evaluator.outcome_classes});
    const torch::Tensor flat_labels =
        labels.unsqueeze(1).expand({labels.size(0), horizon_mask.size(1)}).reshape({-1});
    const torch::Tensor flat_mask = horizon_mask.reshape({-1});
    if (flat_mask.sum().item<std::int64_t>() <= 0) {
      continue;
    }
    evaluator_optimizer_.zero_grad();
    auto options = torch::nn::functional::CrossEntropyFuncOptions().weight(class_weights);
    const torch::Tensor loss = torch::nn::functional::cross_entropy(
        flat_logits.index({flat_mask}),
        flat_labels.index({flat_mask}),
        options);
    loss.backward();
    torch::nn::utils::clip_grad_norm_(evaluator_->parameters(), config_.future_evaluator.max_grad_norm);
    evaluator_optimizer_.step();
    const auto active = flat_mask.sum().item<std::int64_t>();
    total_loss += loss.item<double>() * static_cast<double>(active);
    total_samples += active;
  }
  if (metrics != nullptr && total_samples > 0) {
    metrics->evaluator_loss = total_loss / static_cast<double>(total_samples);
  }
}

void LFPOTrainer::update_target_evaluator(int update_index) {
  if (config_.lfpo.evaluator_target_update_interval <= 0 ||
      update_index % config_.lfpo.evaluator_target_update_interval != 0) {
    return;
  }
  target_evaluator_ = clone_future_evaluator(evaluator_, device_);
  target_evaluator_->eval();
  freeze_parameters(target_evaluator_);
  evaluator_target_update_index_ = update_index;
}

TrainerMetrics LFPOTrainer::update_actor() {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  const RolloutFutureTargets targets =
      build_rollout_targets(rollout_, config_, evaluator_normalizer_, target_evaluator_, device_);
  const int seq_len = std::max(1, config_.lfpo.sequence_length);
  const int agents_per_batch = std::max(1, config_.lfpo.minibatch_size / seq_len);
  const int total_agents = rollout_.num_agents();
  std::int64_t metric_steps = 0;

  for (int epoch = 0; epoch < config_.lfpo.update_epochs; ++epoch) {
    const torch::Tensor perm = torch::randperm(total_agents, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    for (int agent_offset = 0; agent_offset < total_agents; agent_offset += agents_per_batch) {
      const int count = std::min(agents_per_batch, total_agents - agent_offset);
      const torch::Tensor agent_indices = perm.narrow(0, agent_offset, count);
      const torch::Tensor agent_indices_device = agent_indices.to(device_);
      ContinuumState state = state_to_device(rollout_.initial_state_for_agents(agent_indices), device_);
      for (int seq_start = 0; seq_start < rollout_.rollout_length(); seq_start += seq_len) {
        const int chunk_start = seq_start;
        const int chunk_end = std::min(rollout_.rollout_length(), chunk_start + seq_len);
        const int chunk_steps = chunk_end - chunk_start;
        const int burn = seq_start == 0 ? std::min(std::max(0, config_.lfpo.burn_in), chunk_steps) : 0;
        const int loss_start = chunk_start + burn;
        const int loss_steps = chunk_steps - burn;
        const torch::Tensor obs =
            rollout_.obs.narrow(0, chunk_start, chunk_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor episode_starts =
            rollout_.episode_starts.narrow(0, chunk_start, chunk_steps).index_select(1, agent_indices).to(device_);

        const auto forward_start = std::chrono::steady_clock::now();
        ActorSequenceOutput output = actor_->forward_sequence(obs, std::move(state), episode_starts);
        state = detach_state(std::move(output.final_state));
        if (loss_steps <= 0) {
          continue;
        }
        torch::Tensor policy_logits = output.policy_logits.narrow(0, burn, loss_steps);
        torch::Tensor features = output.features.narrow(0, burn, loss_steps);
        const torch::Tensor action_masks =
            rollout_.action_masks.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_).to(torch::kBool);
        const torch::Tensor learner_active =
            rollout_.learner_active.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor candidate_actions =
            rollout_.candidate_actions.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor old_log_probs =
            rollout_.candidate_log_probs.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor executed_actions =
            rollout_.executed_actions.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor target_embeddings =
            targets.embeddings.narrow(0, loss_start, loss_steps).index_select(1, agent_indices_device);
        const torch::Tensor target_horizon_mask =
            targets.horizon_mask.narrow(0, loss_start, loss_steps).index_select(1, agent_indices_device);
        const torch::Tensor target_known =
            targets.known_mask.narrow(0, loss_start, loss_steps).index_select(1, agent_indices_device);

        const auto samples = loss_steps * count;
        const torch::Tensor flat_logits = policy_logits.reshape({samples, config_.model.action_dim});
        const torch::Tensor flat_masks = action_masks.reshape({samples, config_.model.action_dim});
        const torch::Tensor flat_features = features.reshape({samples, actor_->feature_dim()});
        const torch::Tensor flat_candidate_actions = candidate_actions.reshape({samples, config_.lfpo.candidate_count});
        const torch::Tensor flat_old_log_probs = old_log_probs.reshape({samples, config_.lfpo.candidate_count});
        const torch::Tensor flat_active = learner_active.reshape({samples}) > 0.5F;
        if (flat_active.sum().item<std::int64_t>() == 0) {
          continue;
        }

        const torch::Tensor active_logits = flat_logits.index({flat_active});
        const torch::Tensor active_masks = flat_masks.index({flat_active});
        const torch::Tensor active_features = flat_features.index({flat_active});
        const torch::Tensor active_candidate_actions = flat_candidate_actions.index({flat_active});
        const torch::Tensor active_old_log_probs = flat_old_log_probs.index({flat_active});

        const torch::Tensor feature_candidates =
            active_features.unsqueeze(1)
                .expand({active_features.size(0), config_.lfpo.candidate_count, actor_->feature_dim()})
                .reshape({-1, actor_->feature_dim()});
        const torch::Tensor flat_candidate_for_pred = active_candidate_actions.reshape({-1});
        const torch::Tensor predicted = actor_->predict_future_latents(feature_candidates, flat_candidate_for_pred)
                                            .reshape({active_features.size(0), config_.lfpo.candidate_count,
                                                      config_.model.future_horizon_count, config_.model.future_latent_dim});
        const torch::Tensor outcome_logits = target_evaluator_->classify_embeddings(predicted);
        const torch::Tensor candidate_scores = latent_action_scores(outcome_logits);
        const torch::Tensor advantages = relative_candidate_advantages(candidate_scores);

        const torch::Tensor log_probs =
            torch::log_softmax(apply_action_mask_to_logits(active_logits, active_masks), -1);
        const torch::Tensor current_log_probs =
            log_probs.gather(1, active_candidate_actions);
        torch::Tensor policy_loss =
            clipped_lfpo_policy_loss(current_log_probs, active_old_log_probs, advantages, config_.lfpo.clip_range);
        const torch::Tensor entropy = masked_action_entropy(active_logits, active_masks).mean();

        torch::Tensor latent_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        const torch::Tensor flat_executed = executed_actions.reshape({samples}).index({flat_active});
        const torch::Tensor flat_target_embeddings =
            target_embeddings.reshape({samples, config_.model.future_horizon_count, config_.model.future_latent_dim})
                .index({flat_active});
        const torch::Tensor flat_target_horizon_mask =
            target_horizon_mask.reshape({samples, config_.model.future_horizon_count}).index({flat_active});
        const torch::Tensor flat_target_known = target_known.reshape({samples}).index({flat_active});
        if (flat_target_known.sum().item<std::int64_t>() > 0) {
          const torch::Tensor latent_pred = actor_->predict_future_latents(active_features, flat_executed);
          const torch::Tensor mask = flat_target_horizon_mask.unsqueeze(-1).to(torch::kFloat32) *
                                     flat_target_known.view({-1, 1, 1}).to(torch::kFloat32);
          latent_loss = ((latent_pred - flat_target_embeddings).pow(2) * mask).sum() /
                        mask.sum().clamp_min(1.0).mul(config_.model.future_latent_dim);
        }

        const torch::Tensor loss =
            policy_loss + config_.lfpo.latent_loss_coef * latent_loss - config_.lfpo.entropy_coef * entropy;
        metrics.lfpo_forward_backward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - forward_start).count();
        const auto optim_start = std::chrono::steady_clock::now();
        actor_optimizer_.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(actor_->parameters(), config_.lfpo.max_grad_norm);
        actor_optimizer_.step();
        metrics.optimizer_step_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - optim_start).count();
        metrics.policy_loss += policy_loss.item<double>() * active_features.size(0);
        metrics.latent_loss += latent_loss.item<double>() * active_features.size(0);
        metrics.entropy += entropy.item<double>() * active_features.size(0);
        metric_steps += active_features.size(0);
      }
    }
  }
  if (metric_steps > 0) {
    metrics.policy_loss /= static_cast<double>(metric_steps);
    metrics.latent_loss /= static_cast<double>(metric_steps);
    metrics.entropy /= static_cast<double>(metric_steps);
  }
  metrics.update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  return metrics;
}

TrainerMetrics LFPOTrainer::run_update(std::int64_t* global_step, int update_index) {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  CollectorTimings collector_timings{};
  std::int64_t collected_agent_steps = 0;

  const auto collection_start = std::chrono::steady_clock::now();
  rollout_.set_initial_state(collection_state_);
  for (int step = 0; step < config_.lfpo.rollout_length; ++step) {
    torch::Tensor raw_obs_host = collector_->host_observations();
    torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
    torch::Tensor episode_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    torch::Tensor action_masks = collector_->host_action_masks().to(device_, use_pinned_host_buffers_).to(torch::kBool);
    torch::Tensor learner_active = collector_->host_learner_active().to(device_, use_pinned_host_buffers_);
    torch::Tensor snapshot_ids = collector_->host_snapshot_ids().to(device_, use_pinned_host_buffers_);

    torch::Tensor normalized_obs;
    torch::Tensor candidate_actions;
    torch::Tensor candidate_log_probs;
    torch::Tensor executed_actions;
    const auto policy_start = std::chrono::steady_clock::now();
    {
      torch::NoGradGuard no_grad;
      actor_normalizer_.update(raw_obs);
      normalized_obs = actor_normalizer_.normalize(raw_obs);
      ActorStepOutput output = actor_->forward_step(normalized_obs, std::move(collection_state_), episode_starts);
      collection_state_ = std::move(output.state);
      candidate_actions = sample_candidate_actions(output.policy_logits, action_masks, &candidate_log_probs);
      executed_actions = candidate_actions.select(-1, 0);
    }
    metrics.policy_forward_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

    torch::Tensor actions = executed_actions;
    if (self_play_manager_ && self_play_manager_->has_snapshots()) {
      torch::Tensor opponent_actions;
      self_play_manager_->infer_opponent_actions(
          actor_,
          raw_obs,
          action_masks,
          episode_starts,
          snapshot_ids,
          opponent_collection_state_,
          &opponent_actions,
          &metrics.policy_forward_seconds);
      actions = torch::where(snapshot_ids >= 0, opponent_actions, executed_actions);
    }

    const auto decode_start = std::chrono::steady_clock::now();
    const torch::Tensor action_indices_cpu = actions.contiguous().to(torch::kCPU);
    collector_->step(
        std::span<const std::int64_t>(
            action_indices_cpu.data_ptr<std::int64_t>(),
            static_cast<std::size_t>(action_indices_cpu.numel())),
        &collector_timings);
    metrics.action_decode_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();

    torch::Tensor dones = collector_->host_dones().to(device_, use_pinned_host_buffers_);
    outcome_buffer_->record_step(
        raw_obs_host,
        collector_->host_dones(),
        collector_->host_terminal_outcome_labels());

    rollout_.append(
        step,
        raw_obs_host,
        normalized_obs.to(torch::kCPU),
        episode_starts.to(torch::kCPU),
        action_masks.to(torch::kUInt8).to(torch::kCPU),
        learner_active.to(torch::kCPU),
        executed_actions.to(torch::kCPU),
        candidate_actions.to(torch::kCPU),
        candidate_log_probs.to(torch::kCPU),
        collection_trajectory_ids_,
        dones.to(torch::kCPU),
        collector_->host_terminal_outcome_labels());
    const torch::Tensor done_indices = torch::nonzero(collector_->host_dones() > 0.5F).view({-1});
    if (done_indices.numel() > 0) {
      const auto count = done_indices.size(0);
      const torch::Tensor new_ids =
          torch::arange(
              next_trajectory_id_,
              next_trajectory_id_ + count,
              torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
      collection_trajectory_ids_.index_put_({done_indices}, new_ids);
      next_trajectory_id_ += count;
    }
    collected_agent_steps += learner_active.sum().item<std::int64_t>();
  }
  rollout_.set_final_observation(collector_->host_observations());
  const double collection_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

  TrainerMetrics update_metrics = update_actor();
  metrics.policy_loss = update_metrics.policy_loss;
  metrics.latent_loss = update_metrics.latent_loss;
  metrics.entropy = update_metrics.entropy;
  metrics.update_seconds = update_metrics.update_seconds;
  metrics.lfpo_forward_backward_seconds = update_metrics.lfpo_forward_backward_seconds;
  metrics.optimizer_step_seconds = update_metrics.optimizer_step_seconds;
  update_evaluator_from_self_play(update_index, &metrics);
  update_target_evaluator(update_index);

  metrics.obs_build_seconds = collector_timings.obs_build_seconds;
  metrics.mask_build_seconds = collector_timings.mask_build_seconds;
  metrics.env_step_seconds = collector_timings.env_step_seconds;
  metrics.done_reset_seconds = collector_timings.done_reset_seconds;
  metrics.collection_agent_steps_per_second =
      collected_agent_steps > 0 ? static_cast<double>(collected_agent_steps) / collection_seconds : 0.0;
  metrics.update_agent_steps_per_second =
      collected_agent_steps > 0 ? static_cast<double>(collected_agent_steps) / std::max(metrics.update_seconds, 1.0e-9) : 0.0;
  if (global_step != nullptr) {
    *global_step += collected_agent_steps;
  }
  const std::int64_t effective_global_step = global_step != nullptr ? *global_step : collected_agent_steps;
  if (self_play_manager_) {
    const SelfPlayMetrics self_play_metrics =
        self_play_manager_->on_update(actor_, actor_normalizer_, effective_global_step, update_index);
    metrics.self_play_eval_seconds = self_play_metrics.eval_seconds;
    metrics.elo_ratings = self_play_metrics.ratings;
  }
  metrics.overall_agent_steps_per_second =
      collected_agent_steps > 0
          ? static_cast<double>(collected_agent_steps) /
                std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count(), 1.0e-9)
          : 0.0;
  metrics.online_outcome_samples = outcome_buffer_->sample_count();
  metrics.online_outcome_trajectories = outcome_buffer_->trajectories_written();
  metrics.evaluator_target_update_index = evaluator_target_update_index_;
  return metrics;
}

CheckpointMetadata LFPOTrainer::make_checkpoint_metadata(std::int64_t global_step, int update_index) const {
  return CheckpointMetadata{
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_hash(config_.action_table),
      .architecture_name = "lfpo_continuum",
      .device = config_.lfpo.device,
      .global_step = global_step,
      .update_index = update_index,
      .future_evaluator_checkpoint = "future_evaluator",
      .future_evaluator_config_hash = hash_string(nlohmann::json(config_.future_evaluator).dump()),
      .future_evaluator_global_step = outcome_buffer_ ? outcome_buffer_->sample_count() : 0,
      .future_evaluator_update_index = update_index,
      .future_evaluator_target_update_index = evaluator_target_update_index_,
  };
}

void LFPOTrainer::save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index) const {
  std::filesystem::create_directories(directory / "future_evaluator");
  save_experiment_config(config_, (directory / "config.json").string());
  save_checkpoint_metadata(make_checkpoint_metadata(global_step, update_index), (directory / "metadata.json").string());
  torch::serialize::OutputArchive actor_archive;
  actor_->save(actor_archive);
  actor_normalizer_.save(actor_archive);
  actor_archive.save_to((directory / "model.pt").string());
  torch::serialize::OutputArchive actor_optimizer_archive;
  actor_optimizer_.save(actor_optimizer_archive);
  actor_optimizer_archive.save_to((directory / "actor_optimizer.pt").string());

  torch::serialize::OutputArchive target_archive;
  target_evaluator_->save(target_archive);
  evaluator_normalizer_.save(target_archive);
  target_archive.save_to((directory / "future_evaluator" / "model.pt").string());

  torch::serialize::OutputArchive evaluator_archive;
  evaluator_->save(evaluator_archive);
  evaluator_archive.save_to((directory / "future_evaluator" / "online_model.pt").string());

  torch::serialize::OutputArchive evaluator_optimizer_archive;
  evaluator_optimizer_.save(evaluator_optimizer_archive);
  evaluator_optimizer_archive.save_to((directory / "future_evaluator" / "optimizer.pt").string());
}

void LFPOTrainer::prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const {
  const int max_checkpoints = config_.lfpo.max_rolling_checkpoints;
  if (max_checkpoints <= 0 || !std::filesystem::exists(checkpoint_dir)) {
    return;
  }
  std::vector<std::pair<int, std::filesystem::path>> updates;
  for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir)) {
    if (!entry.is_directory()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.rfind("update_", 0) != 0) {
      continue;
    }
    try {
      updates.emplace_back(std::stoi(name.substr(7)), entry.path());
    } catch (...) {
    }
  }
  std::sort(updates.begin(), updates.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
  for (std::size_t i = static_cast<std::size_t>(max_checkpoints); i < updates.size(); ++i) {
    std::filesystem::remove_all(updates[i].second);
  }
}

void LFPOTrainer::train(int updates, const std::string& checkpoint_dir, const std::string& config_path) {
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "lfpo_train");
  std::int64_t global_step = resumed_global_step_;
  const int max_updates = updates <= 0 ? std::numeric_limits<int>::max() : updates;
  for (int index = 0; index < max_updates; ++index) {
    const int update_index = static_cast<int>(resumed_update_index_) + index + 1;
    TrainerMetrics metrics = run_update(&global_step, update_index);
    append_metrics_line(checkpoint_dir, update_index, global_step, metrics);
    std::cout << "update=" << update_index
              << " global_step=" << global_step
              << " policy_loss=" << metrics.policy_loss
              << " latent_loss=" << metrics.latent_loss
              << " evaluator_loss=" << metrics.evaluator_loss
              << " entropy=" << metrics.entropy
              << '\n';
    if (wandb.enabled()) {
      wandb.log(nlohmann::json{
          {"update", update_index},
          {"global_step", global_step},
          {"policy_loss", metrics.policy_loss},
          {"latent_loss", metrics.latent_loss},
          {"evaluator_loss", metrics.evaluator_loss},
          {"entropy", metrics.entropy},
          {"online_outcome_samples", metrics.online_outcome_samples},
          {"online_outcome_trajectories", metrics.online_outcome_trajectories},
      });
    }
    if (config_.lfpo.checkpoint_interval > 0 && update_index % config_.lfpo.checkpoint_interval == 0) {
      save_checkpoint(std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(update_index)), global_step, update_index);
      prune_old_checkpoints(checkpoint_dir);
    }
  }
  save_checkpoint(std::filesystem::path(checkpoint_dir) / "final", global_step, static_cast<int>(resumed_update_index_) + max_updates);
  wandb.finish();
}

}  // namespace pulsar

#endif
