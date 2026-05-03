#include "pulsar/training/appo_trainer.hpp"

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
#include "pulsar/training/cuda_utils.hpp"
#include "pulsar/training/ppo_math.hpp"

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
      config.ppo.rollout_length,
      num_agents,
      config.model.observation_dim,
      action_dim,
      torch::Device(torch::kCPU));
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
      {"value_loss", metrics.value_loss},
      {"entropy", metrics.entropy},
      {"adaptive_epsilon", metrics.adaptive_epsilon},
      {"critic_variance", metrics.critic_variance},
      {"mean_confidence_weight", metrics.mean_confidence_weight},
      {"extrinsic_reward_mean", metrics.extrinsic_reward_mean},
      {"curiosity_reward_mean", metrics.curiosity_reward_mean},
      {"learning_progress_reward_mean", metrics.learning_progress_reward_mean},
      {"bc_regularization_beta", metrics.bc_regularization_beta},
      {"novelty_ema", metrics.novelty_ema},
      {"learning_progress_ema", metrics.learning_progress_ema},
      {"obs_build_seconds", metrics.obs_build_seconds},
      {"mask_build_seconds", metrics.mask_build_seconds},
      {"policy_forward_seconds", metrics.policy_forward_seconds},
      {"action_decode_seconds", metrics.action_decode_seconds},
      {"env_step_seconds", metrics.env_step_seconds},
      {"done_reset_seconds", metrics.done_reset_seconds},
      {"forward_backward_seconds", metrics.forward_backward_seconds},
      {"optimizer_step_seconds", metrics.optimizer_step_seconds},
      {"self_play_eval_seconds", metrics.self_play_eval_seconds},
  };
  for (const auto& [mode, rating] : metrics.elo_ratings) {
    line["elo_" + mode] = rating;
  }
  for (const auto& [head, loss] : metrics.value_losses) {
    line["value_loss_" + head] = loss;
  }
  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

}  // namespace

APPOTrainer::APPOTrainer(
    ExperimentConfig config,
    std::unique_ptr<BatchedRocketSimCollector> collector,
    std::unique_ptr<SelfPlayManager> self_play_manager,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : config_(std::move(config)),
      collector_(std::move(collector)),
      self_play_manager_(std::move(self_play_manager)),
      action_table_(config_.action_table),
      actor_(PPOActor(config_.model)),
      actor_normalizer_(config_.model.observation_dim),
      actor_optimizer_(actor_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate)),
      rollout_(make_rollout_storage(
          config_,
          static_cast<int>(collector_->total_agents()),
          collector_->action_dim())),
      device_(resolve_runtime_device(config_.ppo.device)),
      run_output_root_(std::move(run_output_root)),
      log_initialization_(log_initialization) {
  if (!collector_) {
    throw std::invalid_argument("APPOTrainer requires a collector.");
  }
  total_agents_ = collector_->total_agents();
  collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  opponent_collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  configure_cuda_runtime(device_);
  use_pinned_host_buffers_ = device_.is_cuda();
  actor_->to(device_);
  actor_normalizer_.to(device_);

  head_weights_["extrinsic"] = config_.weight_schedule.initial_extrinsic_weight;
  head_weights_["curiosity"] = config_.weight_schedule.initial_curiosity_weight;
  head_weights_["learning_progress"] = config_.weight_schedule.initial_learning_progress_weight;
  head_weights_["controllability"] = config_.weight_schedule.initial_controllability_weight;

  current_beta_ = config_.bc_regularization.initial_beta;
  novelty_ema_ = torch::zeros({static_cast<std::int64_t>(total_agents_)},
                               torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  learning_progress_ema_ = torch::zeros({static_cast<std::int64_t>(total_agents_)},
                                         torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  has_prev_intrinsic_step_ = torch::zeros({static_cast<std::int64_t>(total_agents_)},
                                           torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

  maybe_initialize_from_checkpoint();

  if (self_play_manager_ && self_play_manager_->enabled()) {
    collector_->set_self_play_assignment_fn(
        [this](std::size_t env_idx, std::uint64_t seed) {
          return self_play_manager_->sample_assignment(env_idx, seed);
        });
  }
}

torch::Tensor APPOTrainer::map_outcome_labels_to_rewards(const torch::Tensor& labels) const {
  torch::Tensor rewards = torch::zeros_like(labels, torch::TensorOptions().dtype(torch::kFloat32));
  rewards.masked_fill_(labels == 0, config_.outcome.score);
  rewards.masked_fill_(labels == 1, config_.outcome.concede);
  return rewards;
}

void APPOTrainer::maybe_initialize_from_checkpoint() {
  if (config_.ppo.init_checkpoint.empty()) {
    return;
  }
  const std::filesystem::path base(config_.ppo.init_checkpoint);
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
  if (log_initialization_) {
    std::cout << "initialized_dappo_from_checkpoint=" << base.string() << '\n';
  }
}


void APPOTrainer::decay_bc_beta() {
  current_beta_ = std::max(current_beta_ * config_.bc_regularization.beta_decay, config_.bc_regularization.min_beta);
}

void APPOTrainer::update_weight_schedule() {
  head_weights_["extrinsic"] = advance_weight_schedule(
      head_weights_["extrinsic"],
      config_.weight_schedule.extrinsic_weight_growth_rate,
      config_.weight_schedule.max_extrinsic_weight);
  head_weights_["curiosity"] = std::max(
      head_weights_["curiosity"] * config_.weight_schedule.intrinsic_weight_decay_rate,
      config_.weight_schedule.min_intrinsic_weight);
  head_weights_["learning_progress"] = std::max(
      head_weights_["learning_progress"] * config_.weight_schedule.intrinsic_weight_decay_rate,
      config_.weight_schedule.min_intrinsic_weight);
}

TrainerMetrics APPOTrainer::update_actor() {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  const int seq_len = std::max(1, config_.ppo.sequence_length);
  const int agents_per_batch = std::max(1, config_.ppo.minibatch_size / seq_len);
  const int total_agents = rollout_.num_agents();
  std::int64_t metric_steps = 0;
  double accumulated_variance = 0.0;
  double accumulated_epsilon = 0.0;
  double accumulated_confidence = 0.0;

  const auto& all_values = rollout_.all_values();
  const auto& all_rewards = rollout_.all_rewards();

  std::unordered_map<std::string, torch::Tensor> per_head_advantages =
      compute_per_head_advantages(all_values, all_rewards, rollout_.dones,
                                   config_.ppo.gamma, config_.ppo.gae_lambda,
                                   rollout_.final_values());
  std::unordered_map<std::string, torch::Tensor> per_head_returns =
      compute_per_head_returns(per_head_advantages, all_values);

  torch::Tensor active_mask = rollout_.learner_active > 0.5F;
  std::unordered_map<std::string, torch::Tensor> normalized_advantages;
  for (const auto& [name, adv] : per_head_advantages) {
    normalized_advantages[name] = normalize_advantage(adv, active_mask);
  }

  torch::Tensor mixed_advantages = mix_advantages(normalized_advantages, head_weights_, active_mask);

  const torch::Tensor atom_support_ext = actor_->value_support("extrinsic").to(device_);

  for (int epoch = 0; epoch < config_.ppo.update_epochs; ++epoch) {
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
        const int burn = seq_start == 0 ? std::min(std::max(0, config_.ppo.burn_in), chunk_steps) : 0;
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

        const torch::Tensor action_masks =
            rollout_.action_masks.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_).to(torch::kBool);
        const torch::Tensor learner_active =
            rollout_.learner_active.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor old_actions =
            rollout_.actions.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor old_log_probs =
            rollout_.action_log_probs.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor chunk_advantages =
            mixed_advantages.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);

        const auto samples = loss_steps * count;
        const torch::Tensor flat_active = learner_active.reshape({samples}) > 0.5F;
        if (flat_active.sum().item<std::int64_t>() == 0) {
          continue;
        }

        torch::Tensor flat_logits = policy_logits.reshape({samples, config_.model.action_dim});
        torch::Tensor flat_masks = action_masks.reshape({samples, config_.model.action_dim});
        torch::Tensor flat_actions = old_actions.reshape({samples});
        torch::Tensor flat_old_log_probs = old_log_probs.reshape({samples});
        torch::Tensor flat_advantages = chunk_advantages.reshape({samples});

        const torch::Tensor active_logits = flat_logits.index({flat_active});
        const torch::Tensor active_masks = flat_masks.index({flat_active});
        const torch::Tensor active_actions = flat_actions.index({flat_active});
        const torch::Tensor active_old_log_probs = flat_old_log_probs.index({flat_active});
        const torch::Tensor active_advantages = flat_advantages.index({flat_active});

        const torch::Tensor log_probs =
            torch::log_softmax(apply_action_mask_to_logits(active_logits, active_masks), -1);
        const torch::Tensor current_log_probs = log_probs.gather(1, active_actions.unsqueeze(1)).squeeze(1);

        float epsilon = config_.ppo.clip_range;
        torch::Tensor confidence_weights = torch::ones({active_advantages.size(0)}, active_advantages.options());
        if (config_.ppo.use_adaptive_epsilon || config_.ppo.use_confidence_weighting) {
          torch::Tensor ext_value_logits_chunk = output.value_ext.logits.narrow(0, burn, loss_steps);
          torch::Tensor flat_ext_value_logits = ext_value_logits_chunk.reshape({samples, config_.model.value_num_atoms});
          torch::Tensor active_ext_value_logits = flat_ext_value_logits.index({flat_active});
          const torch::Tensor critic_variance = compute_distribution_variance(active_ext_value_logits, atom_support_ext);
          if (config_.ppo.use_adaptive_epsilon) {
            epsilon = compute_adaptive_epsilon(
                critic_variance,
                config_.ppo.clip_range,
                config_.ppo.adaptive_epsilon_beta,
                config_.ppo.epsilon_min,
                config_.ppo.epsilon_max);
          }
          if (config_.ppo.use_confidence_weighting) {
            confidence_weights = compute_confidence_weights(
                active_ext_value_logits,
                atom_support_ext,
                config_.ppo.confidence_weight_type,
                config_.ppo.confidence_weight_delta,
                config_.ppo.normalize_confidence_weights);
          }
          accumulated_variance += critic_variance.mean().item<double>() * static_cast<double>(active_logits.size(0));
          accumulated_confidence += confidence_weights.mean().item<double>() * static_cast<double>(active_logits.size(0));
        }
        accumulated_epsilon += static_cast<double>(epsilon) * static_cast<double>(active_logits.size(0));

        torch::Tensor policy_loss =
            clipped_ppo_policy_loss(current_log_probs, active_old_log_probs, active_advantages, epsilon);
        policy_loss = (policy_loss * confidence_weights).mean();

        if (current_beta_ > 0.0F) {
          torch::Tensor bc_loss = torch::nn::functional::cross_entropy(
              active_logits,
              active_actions.to(torch::kLong),
              torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
          policy_loss = policy_loss + current_beta_ * bc_loss;
        }

        const torch::Tensor entropy = masked_action_entropy(active_logits, active_masks).mean();

        torch::Tensor total_value_loss = torch::zeros({}, active_advantages.options());
        const std::vector<std::string> trainable_heads = {"extrinsic", "curiosity", "learning_progress"};

        for (const auto& head_name : trainable_heads) {
          auto returns_it = per_head_returns.find(head_name);
          if (returns_it == per_head_returns.end()) {
            continue;
          }
          torch::Tensor flat_head_returns =
              returns_it->second.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_).reshape({samples});
          torch::Tensor active_head_returns = flat_head_returns.index({flat_active});

          const ValueHeadOutput& head_output = actor_->value_head_output(head_name, output);
          torch::Tensor head_logits_chunk = head_output.logits.narrow(0, burn, loss_steps);
          torch::Tensor flat_head_logits = head_logits_chunk.reshape({samples, head_output.support.size(0)});
          torch::Tensor active_head_logits = flat_head_logits.index({flat_active});

          torch::Tensor head_loss = distributional_value_loss(
              active_head_logits, active_head_returns, head_output.support,
              config_.model.value_v_min, config_.model.value_v_max, config_.model.value_num_atoms);
          total_value_loss = total_value_loss + head_loss;
          metrics.value_losses[head_name] += head_loss.item<double>() * static_cast<double>(active_logits.size(0));
        }

        const torch::Tensor loss =
            policy_loss + config_.ppo.value_coef * total_value_loss - config_.ppo.entropy_coef * entropy;

        metrics.forward_backward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - forward_start).count();
        const auto optim_start = std::chrono::steady_clock::now();
        actor_optimizer_.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(actor_->parameters(), config_.ppo.max_grad_norm);
        actor_optimizer_.step();
        metrics.optimizer_step_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - optim_start).count();

        const auto active_samples = active_logits.size(0);
        metrics.policy_loss += policy_loss.item<double>() * static_cast<double>(active_samples);
        metrics.value_loss += total_value_loss.item<double>() * static_cast<double>(active_samples);
        metrics.entropy += entropy.item<double>() * static_cast<double>(active_samples);
        metric_steps += active_samples;
      }
    }
  }

  if (metric_steps > 0) {
    metrics.policy_loss /= static_cast<double>(metric_steps);
    metrics.value_loss /= static_cast<double>(metric_steps);
    metrics.entropy /= static_cast<double>(metric_steps);
    metrics.adaptive_epsilon = accumulated_epsilon / static_cast<double>(metric_steps);
    metrics.critic_variance = accumulated_variance / static_cast<double>(metric_steps);
    metrics.mean_confidence_weight = accumulated_confidence / static_cast<double>(metric_steps);
    for (auto& [name, loss] : metrics.value_losses) {
      loss /= static_cast<double>(metric_steps);
    }
  }
  metrics.update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  return metrics;
}

TrainerMetrics APPOTrainer::run_update(std::int64_t* global_step, int update_index) {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  CollectorTimings collector_timings{};
  std::int64_t collected_agent_steps = 0;

  const auto collection_start = std::chrono::steady_clock::now();
  rollout_.set_initial_state(collection_state_);
  const torch::Tensor atom_support_ext = actor_->value_support("extrinsic").to(device_);

  double total_extrinsic_reward = 0.0;
  double total_curiosity_reward = 0.0;
  double total_learning_progress_reward = 0.0;
  int64_t total_steps = 0;

  for (int step = 0; step < config_.ppo.rollout_length; ++step) {
    torch::Tensor raw_obs_host = collector_->host_observations();
    torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
    torch::Tensor episode_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    torch::Tensor action_masks = collector_->host_action_masks().to(device_, use_pinned_host_buffers_).to(torch::kBool);
    torch::Tensor learner_active = collector_->host_learner_active().to(device_, use_pinned_host_buffers_);
    torch::Tensor snapshot_ids = collector_->host_snapshot_ids().to(device_, use_pinned_host_buffers_);

    torch::Tensor normalized_obs;
    torch::Tensor actions;
    torch::Tensor action_log_probs;
    ActorStepOutput output;
    const auto policy_start = std::chrono::steady_clock::now();
    {
      torch::NoGradGuard no_grad;
      actor_normalizer_.update(raw_obs);
      normalized_obs = actor_normalizer_.normalize(raw_obs);
      output = actor_->forward_step(normalized_obs, std::move(collection_state_), episode_starts);
      collection_state_ = std::move(output.state);
      actions = sample_masked_actions(output.policy_logits, action_masks, false, &action_log_probs);
    }
    metrics.policy_forward_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

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
      actions = torch::where(snapshot_ids >= 0, opponent_actions, actions);
    }

    torch::Tensor sampled_ext_value = sample_quantile_value(output.value_ext.logits, atom_support_ext);

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
    torch::Tensor terminal_labels = collector_->host_terminal_outcome_labels();
    torch::Tensor extrinsic_rewards = map_outcome_labels_to_rewards(terminal_labels);
    extrinsic_rewards = extrinsic_rewards.to(device_, use_pinned_host_buffers_) * dones;

    torch::Tensor curiosity_rewards = torch::zeros_like(extrinsic_rewards);
    torch::Tensor learning_progress_rewards = torch::zeros_like(extrinsic_rewards);
    torch::Tensor controllability_rewards = torch::zeros_like(extrinsic_rewards);

    // Reset intrinsic memory for agents starting a new episode *before* the reward
    // calculation so we do not compute cross-episode prediction errors.
    torch::Tensor episode_starts_cpu = episode_starts.to(torch::kCPU).to(torch::kBool);
    has_prev_intrinsic_step_.masked_fill_(episode_starts_cpu, false);

    {
      torch::NoGradGuard no_grad;
      // Use per-agent persistent memory: only compute intrinsic rewards for agents
      // that have a valid previous step.
      torch::Tensor has_prev_device = has_prev_intrinsic_step_.to(device_);
      torch::Tensor any_has_prev = has_prev_device.any();
      if (any_has_prev.item<bool>()) {
        torch::Tensor encoded_now = output.encoded;
        torch::Tensor predicted_next = actor_->forward_predict_next(
            prev_intrinsic_encoded_.to(device_), prev_intrinsic_action_.to(device_));
        torch::Tensor pred_error = torch::mse_loss(
            predicted_next, encoded_now, torch::Reduction::None).mean(-1);

        // Only compute rewards where has_prev_step is true; zero elsewhere.
        torch::Tensor mask = has_prev_device.to(torch::kFloat32);
        pred_error = pred_error * mask;

        // Per-agent EMA update: pred_error has shape [num_agents].
        torch::Tensor pred_error_cpu = pred_error.to(torch::kCPU).to(torch::kFloat64);
        torch::Tensor novelty_delta = pred_error_cpu - novelty_ema_;
        // Only update EMA for agents that have a previous step.
        double alpha_novel = static_cast<double>(config_.intrinsic_rewards.novelty_ema_decay);
        double alpha_learn = static_cast<double>(config_.intrinsic_rewards.learning_progress_ema_decay);
        novelty_ema_ = torch::where(
            has_prev_intrinsic_step_,
            alpha_novel * novelty_ema_ + (1.0 - alpha_novel) * pred_error_cpu,
            novelty_ema_);
        learning_progress_ema_ = torch::where(
            has_prev_intrinsic_step_,
            alpha_learn * learning_progress_ema_ + (1.0 - alpha_learn) * novelty_delta.abs(),
            learning_progress_ema_);

        curiosity_rewards = pred_error * config_.intrinsic_rewards.curiosity_weight;

        // Learning progress rewards are per-agent, shaped by the per-agent EMA.
        learning_progress_rewards = learning_progress_ema_.to(device_).to(torch::kFloat32) *
                                     config_.intrinsic_rewards.learning_progress_weight *
                                     mask;

        if (config_.intrinsic_rewards.use_controllability_gate) {
          torch::Tensor prev_enc_dev = prev_intrinsic_encoded_.to(device_);
          torch::Tensor inv_logits = actor_->forward_predict_action(prev_enc_dev, encoded_now);
          torch::Tensor inv_probs = torch::softmax(inv_logits, -1);
          torch::Tensor prev_act_dev = prev_intrinsic_action_.to(device_);
          torch::Tensor correct_probs = inv_probs.gather(1, prev_act_dev.unsqueeze(1)).squeeze(1);
          torch::Tensor controllability = correct_probs * mask;
          curiosity_rewards = curiosity_rewards * controllability;
          learning_progress_rewards = learning_progress_rewards * controllability;
          controllability_rewards = controllability * config_.intrinsic_rewards.controllability_weight;
        }
      }
    }

    total_extrinsic_reward += extrinsic_rewards.sum().item<double>();
    total_curiosity_reward += curiosity_rewards.sum().item<double>();
    total_learning_progress_reward += learning_progress_rewards.sum().item<double>();
    total_steps += extrinsic_rewards.numel();

    // Store current encoded/action for next step's intrinsic reward.
    // Allocate persistent tensors on first use (encoded dim not known until first forward).
    if (!prev_intrinsic_encoded_.defined()) {
      prev_intrinsic_encoded_ = output.encoded.detach().to(torch::kCPU).clone();
      prev_intrinsic_action_ = actions.detach().to(torch::kCPU).clone();
    } else {
      prev_intrinsic_encoded_.copy_(output.encoded.detach().to(torch::kCPU));
      prev_intrinsic_action_.copy_(actions.detach().to(torch::kCPU));
    }
    has_prev_intrinsic_step_.fill_(true);

    const torch::Tensor atom_support_cur = actor_->value_support("curiosity").to(device_);
    const torch::Tensor atom_support_learn = actor_->value_support("learning_progress").to(device_);
    const torch::Tensor atom_support_ctrl = actor_->value_support("controllability").to(device_);

    std::unordered_map<std::string, torch::Tensor> all_values;
    all_values["extrinsic"] = sampled_ext_value.to(torch::kCPU);
    all_values["curiosity"] = compute_mean_value(output.value_cur.logits, atom_support_cur).to(torch::kCPU);
    all_values["learning_progress"] = compute_mean_value(output.value_learn.logits, atom_support_learn).to(torch::kCPU);
    all_values["controllability"] = compute_mean_value(output.value_ctrl.logits, atom_support_ctrl).to(torch::kCPU);

    std::unordered_map<std::string, torch::Tensor> all_rewards;
    all_rewards["extrinsic"] = extrinsic_rewards.to(torch::kCPU);
    all_rewards["curiosity"] = curiosity_rewards.to(torch::kCPU);
    all_rewards["learning_progress"] = learning_progress_rewards.to(torch::kCPU);
    all_rewards["controllability"] = controllability_rewards.to(torch::kCPU);

    rollout_.append(
        step,
        raw_obs_host,
        normalized_obs.to(torch::kCPU),
        episode_starts_cpu,
        action_masks.to(torch::kUInt8).to(torch::kCPU),
        learner_active.to(torch::kCPU),
        actions.to(torch::kCPU),
        action_log_probs.to(torch::kCPU),
        all_values,
        all_rewards,
        dones.to(torch::kCPU));

    collected_agent_steps += learner_active.sum().item<std::int64_t>();

  }
  rollout_.set_final_observation(collector_->host_observations());

  // Compute bootstrap values for GAE from the final observation.
  {
    torch::NoGradGuard no_grad;
    torch::Tensor final_raw_obs = collector_->host_observations().to(device_, use_pinned_host_buffers_);
    torch::Tensor final_normalized = actor_normalizer_.normalize(final_raw_obs);
    torch::Tensor final_starts = torch::ones(
        {static_cast<std::int64_t>(total_agents_)},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    ActorStepOutput final_output = actor_->forward_step(
        final_normalized, std::move(collection_state_), final_starts);
    collection_state_ = std::move(final_output.state);

    const torch::Tensor atom_support_cur = actor_->value_support("curiosity").to(device_);
    const torch::Tensor atom_support_learn = actor_->value_support("learning_progress").to(device_);
    const torch::Tensor atom_support_ctrl = actor_->value_support("controllability").to(device_);

    std::unordered_map<std::string, torch::Tensor> bootstrap_values;
    bootstrap_values["extrinsic"] = compute_mean_value(
        final_output.value_ext.logits, atom_support_ext).to(torch::kCPU);
    bootstrap_values["curiosity"] = compute_mean_value(
        final_output.value_cur.logits, atom_support_cur).to(torch::kCPU);
    bootstrap_values["learning_progress"] = compute_mean_value(
        final_output.value_learn.logits, atom_support_learn).to(torch::kCPU);
    bootstrap_values["controllability"] = compute_mean_value(
        final_output.value_ctrl.logits, atom_support_ctrl).to(torch::kCPU);
    rollout_.set_final_values(bootstrap_values);
  }

  const double collection_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

  if (total_steps > 0) {
    metrics.extrinsic_reward_mean = total_extrinsic_reward / static_cast<double>(total_steps);
    metrics.curiosity_reward_mean = total_curiosity_reward / static_cast<double>(total_steps);
    metrics.learning_progress_reward_mean = total_learning_progress_reward / static_cast<double>(total_steps);
  }
  metrics.novelty_ema = novelty_ema_.mean().item<double>();
  metrics.learning_progress_ema = learning_progress_ema_.mean().item<double>();
  metrics.bc_regularization_beta = static_cast<double>(current_beta_);

  TrainerMetrics update_metrics = update_actor();
  metrics.policy_loss = update_metrics.policy_loss;
  metrics.value_loss = update_metrics.value_loss;
  metrics.entropy = update_metrics.entropy;
  metrics.update_seconds = update_metrics.update_seconds;
  metrics.forward_backward_seconds = update_metrics.forward_backward_seconds;
  metrics.optimizer_step_seconds = update_metrics.optimizer_step_seconds;
  metrics.adaptive_epsilon = update_metrics.adaptive_epsilon;
  metrics.critic_variance = update_metrics.critic_variance;
  metrics.mean_confidence_weight = update_metrics.mean_confidence_weight;
  metrics.value_losses = update_metrics.value_losses;

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

  update_weight_schedule();
  decay_bc_beta();

  metrics.overall_agent_steps_per_second =
      collected_agent_steps > 0
          ? static_cast<double>(collected_agent_steps) /
                std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count(), 1.0e-9)
          : 0.0;
  return metrics;
}

CheckpointMetadata APPOTrainer::make_checkpoint_metadata(std::int64_t global_step, int update_index) const {
  return CheckpointMetadata{
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_hash(config_.action_table),
      .architecture_name = "dappo_continuum",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
      .critic_heads = actor_->enabled_critic_heads(),
  };
}

void APPOTrainer::save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index) const {
  std::filesystem::create_directories(directory);
  save_experiment_config(config_, (directory / "config.json").string());
  save_checkpoint_metadata(make_checkpoint_metadata(global_step, update_index), (directory / "metadata.json").string());
  torch::serialize::OutputArchive actor_archive;
  actor_->save(actor_archive);
  actor_normalizer_.save(actor_archive);
  actor_archive.save_to((directory / "model.pt").string());
  torch::serialize::OutputArchive actor_optimizer_archive;
  actor_optimizer_.save(actor_optimizer_archive);
  actor_optimizer_archive.save_to((directory / "actor_optimizer.pt").string());
}

void APPOTrainer::prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const {
  const int max_checkpoints = config_.ppo.max_rolling_checkpoints;
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

void APPOTrainer::train(int updates, const std::string& checkpoint_dir, const std::string& config_path) {
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "dappo_train");
  std::int64_t global_step = resumed_global_step_;
  const int max_updates = updates <= 0 ? std::numeric_limits<int>::max() : updates;
  for (int index = 0; index < max_updates; ++index) {
    const int update_index = static_cast<int>(resumed_update_index_) + index + 1;
    TrainerMetrics metrics = run_update(&global_step, update_index);
    append_metrics_line(checkpoint_dir, update_index, global_step, metrics);
    std::cout << "update=" << update_index
              << " global_step=" << global_step
              << " policy_loss=" << metrics.policy_loss
              << " value_loss=" << metrics.value_loss
              << " entropy=" << metrics.entropy
              << " epsilon=" << metrics.adaptive_epsilon
              << " critic_var=" << metrics.critic_variance
              << " conf_weight=" << metrics.mean_confidence_weight
              << " ext_r=" << metrics.extrinsic_reward_mean
              << " cur_r=" << metrics.curiosity_reward_mean
              << " learn_r=" << metrics.learning_progress_reward_mean
              << " beta=" << metrics.bc_regularization_beta
              << '\n';
    if (wandb.enabled()) {
      wandb.log(nlohmann::json{
          {"update", update_index},
          {"global_step", global_step},
          {"policy_loss", metrics.policy_loss},
          {"value_loss", metrics.value_loss},
          {"entropy", metrics.entropy},
          {"adaptive_epsilon", metrics.adaptive_epsilon},
          {"critic_variance", metrics.critic_variance},
          {"mean_confidence_weight", metrics.mean_confidence_weight},
          {"extrinsic_reward_mean", metrics.extrinsic_reward_mean},
          {"curiosity_reward_mean", metrics.curiosity_reward_mean},
          {"learning_progress_reward_mean", metrics.learning_progress_reward_mean},
          {"bc_regularization_beta", metrics.bc_regularization_beta},
          {"novelty_ema", metrics.novelty_ema},
      });
    }
    if (config_.ppo.checkpoint_interval > 0 && update_index % config_.ppo.checkpoint_interval == 0) {
      save_checkpoint(std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(update_index)), global_step, update_index);
      prune_old_checkpoints(checkpoint_dir);
    }
  }
  save_checkpoint(std::filesystem::path(checkpoint_dir) / "final", global_step, static_cast<int>(resumed_update_index_) + max_updates);
  wandb.finish();
}

}  // namespace pulsar

#endif
