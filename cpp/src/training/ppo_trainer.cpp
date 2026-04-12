#include "pulsar/training/ppo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <ATen/autocast_mode.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pulsar {
namespace {

class ScopedAutocast {
 public:
  ScopedAutocast(bool enabled, at::DeviceType device_type, at::ScalarType dtype)
      : enabled_(enabled), device_type_(device_type) {
    if (!enabled_) {
      return;
    }
    previous_enabled_ = at::autocast::is_autocast_enabled(device_type_);
    previous_dtype_ = at::autocast::get_autocast_dtype(device_type_);
    previous_cache_enabled_ = at::autocast::is_autocast_cache_enabled();
    at::autocast::set_autocast_enabled(device_type_, true);
    at::autocast::set_autocast_dtype(device_type_, dtype);
    at::autocast::set_autocast_cache_enabled(true);
    at::autocast::increment_nesting();
  }

  ~ScopedAutocast() {
    if (!enabled_) {
      return;
    }
    at::autocast::decrement_nesting();
    at::autocast::set_autocast_enabled(device_type_, previous_enabled_);
    at::autocast::set_autocast_dtype(device_type_, previous_dtype_);
    at::autocast::set_autocast_cache_enabled(previous_cache_enabled_);
  }

 private:
  bool enabled_ = false;
  at::DeviceType device_type_ = at::kCPU;
  bool previous_enabled_ = false;
  bool previous_cache_enabled_ = false;
  at::ScalarType previous_dtype_ = at::kFloat;
};

torch::Tensor gather_state_tensor(const torch::Tensor& tensor, const torch::Tensor& agent_indices) {
  if (!tensor.defined()) {
    return tensor;
  }
  return tensor.index_select(0, agent_indices);
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
    const ExperimentConfig& active_config) {
  if (metadata.schema_version != active_config.schema_version) {
    throw std::runtime_error("Init checkpoint schema_version does not match the active config.");
  }
  if (metadata.obs_schema_version != active_config.obs_schema_version) {
    throw std::runtime_error("Init checkpoint obs_schema_version does not match the active config.");
  }
  if (metadata.action_table_hash != action_table_hash(active_config.action_table)) {
    throw std::runtime_error("Init checkpoint action table hash does not match the active config.");
  }
  if (stable_model_signature(checkpoint_config) != stable_model_signature(active_config)) {
    throw std::runtime_error("Init checkpoint model/value signature does not match the active config.");
  }
}

void validate_aux_checkpoint_compatibility(
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
      {"reward_mean", metrics.reward_mean},
      {"policy_loss", metrics.policy_loss},
      {"value_loss", metrics.value_loss},
      {"entropy", metrics.entropy},
      {"value_entropy", metrics.value_entropy},
      {"value_variance", metrics.value_variance},
      {"obs_build_seconds", metrics.obs_build_seconds},
      {"mask_build_seconds", metrics.mask_build_seconds},
      {"policy_forward_seconds", metrics.policy_forward_seconds},
      {"action_decode_seconds", metrics.action_decode_seconds},
      {"env_step_seconds", metrics.env_step_seconds},
      {"reward_done_seconds", metrics.reward_done_seconds},
      {"ngp_reward_seconds", metrics.ngp_reward_seconds},
      {"rollout_append_seconds", metrics.rollout_append_seconds},
      {"gae_seconds", metrics.gae_seconds},
      {"ppo_forward_backward_seconds", metrics.ppo_forward_backward_seconds},
      {"optimizer_step_seconds", metrics.optimizer_step_seconds},
      {"self_play_eval_seconds", metrics.self_play_eval_seconds},
  };
  for (const auto& [mode, rating] : metrics.elo_ratings) {
    line["elo_" + mode] = rating;
  }

  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

}  // namespace

PPOTrainer::PPOTrainer(
    ExperimentConfig config,
    std::unique_ptr<BatchedRocketSimCollector> collector,
    ActionParserPtr action_parser,
    std::unique_ptr<SelfPlayManager> self_play_manager)
    : config_(std::move(config)),
      collector_(std::move(collector)),
      action_parser_(std::move(action_parser)),
      self_play_manager_(std::move(self_play_manager)),
      action_table_(config_.action_table),
      model_(SharedActorCritic(config_.model, config_.ppo)),
      normalizer_(config_.model.observation_dim),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate)),
      rollout_(
          config_.ppo.rollout_length,
          static_cast<int>(collector_->total_agents()),
          config_.model.observation_dim,
          collector_->action_dim(),
          torch::Device(config_.ppo.device)),
      device_(config_.ppo.device),
      ngp_normalizer_(config_.model.observation_dim) {
  if (!collector_ || !action_parser_) {
    throw std::invalid_argument("PPOTrainer requires a collector and action parser.");
  }

  total_agents_ = collector_->total_agents();
  collection_state_ = model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  opponent_collection_state_ = model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  host_actions_.resize(total_agents_);
  use_pinned_host_buffers_ = device_.is_cuda();
  model_->to(device_);
  normalizer_.to(device_);
  validate_precision_mode();
  maybe_initialize_from_checkpoint();
  maybe_initialize_ngp_reward();

  if (self_play_manager_ && self_play_manager_->enabled()) {
    collector_->set_self_play_assignment_fn(
        [this](std::size_t env_idx, std::uint64_t seed) {
          return self_play_manager_->sample_assignment(env_idx, seed);
        });
  }
}

ContinuumState PPOTrainer::replay_state_until(std::int64_t start_step, const torch::Tensor& agent_indices) {
  ContinuumState state = model_->initial_state(agent_indices.size(0), device_);
  if (start_step <= 0) {
    return state;
  }

  const bool use_amp = config_.ppo.precision.mode == "amp_bf16" && device_.is_cuda();
  torch::NoGradGuard no_grad;
  for (std::int64_t step = 0; step < start_step; ++step) {
    const torch::Tensor obs = rollout_.obs[step].index_select(0, agent_indices);
    const torch::Tensor starts = rollout_.episode_starts[step].index_select(0, agent_indices);
    ScopedAutocast autocast(use_amp, device_.type(), at::kBFloat16);
    SequenceOutput out = model_->forward_sequence(obs.unsqueeze(0), std::move(state), starts.unsqueeze(0));
    state = std::move(out.final_state);
  }
  return state;
}

torch::Tensor PPOTrainer::sample_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs) const {
  return sample_masked_actions(logits, action_masks, deterministic, log_probs);
}

std::vector<std::int64_t> PPOTrainer::actions_to_indices(const torch::Tensor& actions) const {
  std::vector<std::int64_t> action_indices(static_cast<std::size_t>(actions.size(0)));
  const torch::Tensor action_cpu = actions.to(torch::kCPU);
  std::memcpy(action_indices.data(), action_cpu.data_ptr<std::int64_t>(), action_indices.size() * sizeof(std::int64_t));
  return action_indices;
}

torch::Tensor PPOTrainer::categorical_projection(const torch::Tensor& returns) const {
  return categorical_value_projection(
      returns,
      config_.ppo.value_v_min,
      config_.ppo.value_v_max,
      config_.ppo.value_num_atoms);
}

torch::Tensor PPOTrainer::confidence_weights(const torch::Tensor& value_logits) const {
  return compute_confidence_weights(model_, config_.ppo, value_logits);
}

torch::Tensor PPOTrainer::adaptive_epsilon(const torch::Tensor& value_logits) const {
  return compute_adaptive_epsilon(model_, config_.ppo, value_logits);
}

void PPOTrainer::validate_precision_mode() const {
  validate_precision_mode_or_throw(config_.ppo.precision, device_);
}

void PPOTrainer::maybe_initialize_from_checkpoint() {
  if (config_.ppo.init_checkpoint.empty()) {
    return;
  }

  namespace fs = std::filesystem;
  const fs::path base(config_.ppo.init_checkpoint);
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_init_checkpoint_compatibility(metadata, checkpoint_config, config_);

  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string());
  model_->load(archive);
  normalizer_.load(archive);
  model_->to(device_);
  normalizer_.to(device_);

  std::cout << "initialized_from_checkpoint=" << base.string() << '\n';
}

void PPOTrainer::maybe_initialize_ngp_reward() {
  use_shaped_reward_ = config_.reward.mode != "ngp" && config_.reward.shaped_scale != 0.0F;
  use_ngp_reward_ = (config_.reward.mode == "ngp" || config_.reward.mode == "hybrid") &&
                    !config_.reward.ngp_checkpoint.empty();
  if (!use_ngp_reward_) {
    if (config_.reward.mode == "ngp" || config_.reward.mode == "hybrid") {
      throw std::runtime_error("reward.ngp_checkpoint must be set when reward.mode is ngp or hybrid.");
    }
    return;
  }

  namespace fs = std::filesystem;
  const fs::path base(config_.reward.ngp_checkpoint);
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_aux_checkpoint_compatibility(metadata, checkpoint_config, config_, "NGP checkpoint");

  ngp_model_ = SharedActorCritic(config_.model, config_.ppo);
  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string());
  ngp_model_->load(archive);
  ngp_normalizer_.load(archive);
  ngp_model_->to(device_);
  ngp_normalizer_.to(device_);
  ngp_model_->eval();
  ngp_collection_state_ = ngp_model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);

  std::cout << "initialized_ngp_reward_from_checkpoint=" << base.string() << '\n';
}

torch::Tensor PPOTrainer::ngp_scalar(const torch::Tensor& logits) const {
  const torch::Tensor probs = torch::softmax(logits, -1);
  return probs.select(-1, 0) - probs.select(-1, 1);
}

TrainerMetrics PPOTrainer::update_policy() {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  const bool use_amp = config_.ppo.precision.mode == "amp_bf16" && device_.is_cuda();
  const int seq_len = std::max(1, config_.ppo.sequence_length);
  const int burn_in = std::max(0, config_.ppo.burn_in);
  const int agents_per_batch = std::max(1, config_.ppo.minibatch_size / seq_len);
  const auto total_agents = static_cast<int>(rollout_.num_agents());
  std::int64_t metric_steps = 0;

  for (int epoch = 0; epoch < config_.ppo.epochs; ++epoch) {
    const torch::Tensor perm = torch::randperm(total_agents, torch::TensorOptions().dtype(torch::kLong).device(device_));
    for (int agent_offset = 0; agent_offset < total_agents; agent_offset += agents_per_batch) {
      const int count = std::min(agents_per_batch, total_agents - agent_offset);
      const torch::Tensor agent_indices = perm.narrow(0, agent_offset, count);

      for (int seq_start = 0; seq_start < rollout_.rollout_length(); seq_start += seq_len) {
        const int effective_start = seq_start;
        const int context_start = std::max(0, effective_start - burn_in);
        const int effective_end = std::min(rollout_.rollout_length(), effective_start + seq_len);
        const int total_steps = effective_end - context_start;
        const int burn = effective_start - context_start;

        ContinuumState state = replay_state_until(context_start, agent_indices);
        const torch::Tensor obs = rollout_.obs.narrow(0, context_start, total_steps).index_select(1, agent_indices);
        const torch::Tensor episode_starts =
            rollout_.episode_starts.narrow(0, context_start, total_steps).index_select(1, agent_indices);
        const torch::Tensor action_masks =
            rollout_.action_masks.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices).to(torch::kBool);
        const torch::Tensor learner_active =
            rollout_.learner_active.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor actions =
            rollout_.actions.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor old_log_probs =
            rollout_.log_probs.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor returns =
            rollout_.returns.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        torch::Tensor advantages =
            rollout_.advantages.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);

        const auto ppo_start = std::chrono::steady_clock::now();
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-6);
        SequenceOutput output;
        {
          ScopedAutocast autocast(use_amp, device_.type(), at::kBFloat16);
          output = model_->forward_sequence(obs, std::move(state), episode_starts);
        }
        torch::Tensor policy_logits = output.policy_logits.narrow(0, burn, effective_end - effective_start);
        torch::Tensor value_logits = output.value_logits.narrow(0, burn, effective_end - effective_start);

        const torch::Tensor flat_policy_logits = policy_logits.reshape({-1, config_.model.action_dim});
        const torch::Tensor flat_action_masks = action_masks.reshape({-1, config_.model.action_dim});
        const torch::Tensor flat_value_logits = value_logits.reshape({-1, config_.ppo.value_num_atoms});
        const torch::Tensor flat_actions = actions.reshape({-1});
        const torch::Tensor flat_old_log_probs = old_log_probs.reshape({-1});
        const torch::Tensor flat_advantages = advantages.reshape({-1});
        const torch::Tensor flat_returns = returns.reshape({-1});
        const torch::Tensor flat_active = learner_active.reshape({-1}) > 0.5;

        if (flat_active.sum().item<std::int64_t>() == 0) {
          continue;
        }

        const torch::Tensor active_policy_logits = flat_policy_logits.index({flat_active});
        const torch::Tensor active_action_masks = flat_action_masks.index({flat_active});
        const torch::Tensor active_value_logits = flat_value_logits.index({flat_active});
        const torch::Tensor active_actions = flat_actions.index({flat_active});
        const torch::Tensor active_old_log_probs = flat_old_log_probs.index({flat_active});
        const torch::Tensor active_advantages = flat_advantages.index({flat_active});
        const torch::Tensor active_returns = flat_returns.index({flat_active});

        const torch::Tensor active_masked_logits = apply_action_mask_to_logits(active_policy_logits, active_action_masks);
        const torch::Tensor current_log_probs =
            torch::log_softmax(active_masked_logits, -1).gather(-1, active_actions.unsqueeze(-1)).squeeze(-1);
        const torch::Tensor ratio = torch::exp(current_log_probs - active_old_log_probs);
        const torch::Tensor epsilon = adaptive_epsilon(active_value_logits);
        const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - epsilon, 1.0 + epsilon);
        const torch::Tensor weights = confidence_weights(active_value_logits);
        const torch::Tensor unclipped = ratio * active_advantages;
        const torch::Tensor clipped = clipped_ratio * active_advantages;
        const torch::Tensor policy_loss = -(torch::min(unclipped, clipped) * weights).mean();

        const torch::Tensor action_entropy = masked_action_entropy(active_policy_logits, active_action_masks).mean();
        const torch::Tensor target_dist = categorical_projection(active_returns);
        const torch::Tensor critic_loss =
            -(target_dist * torch::log_softmax(active_value_logits, -1)).sum(-1).mean();
        const torch::Tensor loss =
            policy_loss + config_.ppo.value_coef * critic_loss - config_.ppo.entropy_coef * action_entropy;
        metrics.ppo_forward_backward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ppo_start).count();

        const auto optim_start = std::chrono::steady_clock::now();
        optimizer_.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.ppo.max_grad_norm);
        optimizer_.step();
        metrics.optimizer_step_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - optim_start).count();

        metrics.policy_loss += policy_loss.item<double>() * active_actions.size(0);
        metrics.value_loss += critic_loss.item<double>() * active_actions.size(0);
        metrics.entropy += action_entropy.item<double>() * active_actions.size(0);
        metrics.value_entropy += model_->value_entropy(active_value_logits).mean().item<double>() * active_actions.size(0);
        metrics.value_variance += model_->value_variance(active_value_logits).mean().item<double>() * active_actions.size(0);
        metric_steps += active_actions.size(0);
      }
    }
  }

  if (metric_steps > 0) {
    metrics.policy_loss /= static_cast<double>(metric_steps);
    metrics.value_loss /= static_cast<double>(metric_steps);
    metrics.entropy /= static_cast<double>(metric_steps);
    metrics.value_entropy /= static_cast<double>(metric_steps);
    metrics.value_variance /= static_cast<double>(metric_steps);
  }
  metrics.update_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  return metrics;
}

CheckpointMetadata PPOTrainer::make_checkpoint_metadata(
    std::int64_t global_step,
    std::int64_t update_index) const {
  return {
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_.hash(),
      .architecture_name = "continuum_dppo",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
  };
}

void PPOTrainer::save_checkpoint(
    const std::string& checkpoint_dir,
    std::int64_t global_step,
    std::int64_t update_index) {
  namespace fs = std::filesystem;
  const fs::path directory = fs::path(checkpoint_dir) / ("update_" + std::to_string(update_index));
  save_checkpoint_to_directory(directory, global_step, update_index);
}

void PPOTrainer::save_checkpoint_to_directory(
    const std::filesystem::path& directory,
    std::int64_t global_step,
    std::int64_t update_index) {
  namespace fs = std::filesystem;
  fs::remove_all(directory);
  fs::create_directories(directory);

  save_experiment_config(config_, (directory / "config.json").string());
  save_checkpoint_metadata(make_checkpoint_metadata(global_step, update_index), (directory / "metadata.json").string());

  torch::serialize::OutputArchive archive;
  model_->save(archive);
  normalizer_.save(archive);
  archive.save_to((directory / "model.pt").string());

  torch::serialize::OutputArchive optimizer_archive;
  optimizer_.save(optimizer_archive);
  optimizer_archive.save_to((directory / "optimizer.pt").string());
}

void PPOTrainer::train(int updates, const std::string& checkpoint_dir, const std::string& config_path) {
  std::int64_t global_step = 0;
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "ppo_train");
  const bool use_amp = config_.ppo.precision.mode == "amp_bf16" && device_.is_cuda();

  for (int update_index = 0; update_index < updates; ++update_index) {
    const auto update_start = std::chrono::steady_clock::now();
    TrainerMetrics metrics{};
    CollectorTimings collector_timings{};
    std::int64_t collected_agent_steps = 0;

    const auto collection_start = std::chrono::steady_clock::now();
    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      torch::Tensor raw_obs_host = collector_->collect_observations(&collector_timings);
      collector_->collect_action_masks(&collector_timings);

      const torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
      const torch::Tensor episode_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
      const torch::Tensor action_masks = collector_->host_action_masks().to(device_, use_pinned_host_buffers_).to(torch::kBool);
      const torch::Tensor learner_active = collector_->host_learner_active().to(device_, use_pinned_host_buffers_);
      const torch::Tensor snapshot_ids = collector_->host_snapshot_ids().to(device_, use_pinned_host_buffers_);

      torch::Tensor obs = raw_obs;
      torch::Tensor actions;
      torch::Tensor current_actions;
      torch::Tensor log_probs;
      torch::Tensor sampled_values;
      torch::Tensor ngp_prev_scalar;
      ContinuumState next_ngp_state;

      const auto policy_start = std::chrono::steady_clock::now();
      {
        torch::NoGradGuard no_grad;
        ScopedAutocast autocast(use_amp, device_.type(), at::kBFloat16);
        if (use_ngp_reward_) {
          const torch::Tensor normalized_ngp_obs = ngp_normalizer_.normalize(raw_obs);
          PolicyOutput ngp_output =
              ngp_model_->forward_step(normalized_ngp_obs, std::move(ngp_collection_state_), episode_starts);
          ngp_prev_scalar = ngp_scalar(ngp_output.next_goal_logits);
          next_ngp_state = std::move(ngp_output.state);
        }

        normalizer_.update(obs);
        obs = normalizer_.normalize(obs);
        PolicyOutput output = model_->forward_step(obs, std::move(collection_state_), episode_starts);
        collection_state_ = std::move(output.state);
        current_actions = sample_actions(output.policy_logits, action_masks, false, &log_probs);
        sampled_values = output.sampled_values;
      }
      actions = current_actions;
      metrics.policy_forward_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

      if (self_play_manager_ && self_play_manager_->has_snapshots()) {
        torch::Tensor opponent_actions;
        self_play_manager_->infer_opponent_actions(
            model_,
            config_.ppo.precision,
            raw_obs,
            action_masks,
            episode_starts,
            snapshot_ids,
            opponent_collection_state_,
            &opponent_actions,
            &metrics.policy_forward_seconds);
        actions = torch::where(snapshot_ids >= 0, opponent_actions, current_actions);
      }

      const auto decode_start = std::chrono::steady_clock::now();
      const std::vector<std::int64_t> action_indices = actions_to_indices(actions);
      action_parser_->parse_actions_into(action_indices, host_actions_);
      metrics.action_decode_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();

      collector_->step(host_actions_, use_ngp_reward_, &collector_timings);
      torch::Tensor rewards = collector_->host_rewards().to(device_, use_pinned_host_buffers_);
      const torch::Tensor dones = collector_->host_dones().to(device_, use_pinned_host_buffers_);
      if (!use_shaped_reward_) {
        rewards.zero_();
      }
      if (use_ngp_reward_) {
        const auto ngp_start = std::chrono::steady_clock::now();
        torch::NoGradGuard no_grad;
        ScopedAutocast autocast(use_amp, device_.type(), at::kBFloat16);
        const torch::Tensor post_step_obs = collector_->host_post_step_obs().to(device_, use_pinned_host_buffers_);
        const torch::Tensor zero_starts = torch::zeros_like(dones);
        const torch::Tensor normalized_post_step_obs = ngp_normalizer_.normalize(post_step_obs);
        PolicyOutput ngp_output =
            ngp_model_->forward_step(normalized_post_step_obs, std::move(next_ngp_state), zero_starts);
        const torch::Tensor ngp_current_scalar = ngp_scalar(ngp_output.next_goal_logits);
        rewards = rewards + config_.reward.ngp_scale * (ngp_current_scalar - ngp_prev_scalar);
        ngp_collection_state_ = std::move(ngp_output.state);
        metrics.ngp_reward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ngp_start).count();
      }

      const auto append_start = std::chrono::steady_clock::now();
      rollout_.append(
          step,
          obs,
          episode_starts,
          action_masks.to(torch::kUInt8),
          learner_active,
          actions,
          log_probs,
          rewards,
          dones,
          sampled_values);
      metrics.rollout_append_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - append_start).count();
      collected_agent_steps += learner_active.sum().item<std::int64_t>();
    }
    const double collection_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

    const auto gae_start = std::chrono::steady_clock::now();
    const torch::Tensor last_obs_host = collector_->collect_observations(&collector_timings);
    const torch::Tensor last_obs = normalizer_.normalize(last_obs_host.to(device_, use_pinned_host_buffers_));
    const torch::Tensor last_episode_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    torch::Tensor last_sampled_value;
    const auto bootstrap_start = std::chrono::steady_clock::now();
    {
      torch::NoGradGuard no_grad;
      ScopedAutocast autocast(use_amp, device_.type(), at::kBFloat16);
      PolicyOutput output = model_->forward_step(last_obs, collection_state_, last_episode_starts);
      last_sampled_value = output.sampled_values;
    }
    metrics.policy_forward_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - bootstrap_start).count();
    rollout_.compute_returns_and_advantages(last_sampled_value, config_.ppo.gamma, config_.ppo.gae_lambda);
    metrics.gae_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - gae_start).count();

    TrainerMetrics update_metrics = update_policy();
    metrics.policy_loss = update_metrics.policy_loss;
    metrics.value_loss = update_metrics.value_loss;
    metrics.entropy = update_metrics.entropy;
    metrics.value_entropy = update_metrics.value_entropy;
    metrics.value_variance = update_metrics.value_variance;
    metrics.update_seconds = update_metrics.update_seconds;
    metrics.ppo_forward_backward_seconds = update_metrics.ppo_forward_backward_seconds;
    metrics.optimizer_step_seconds = update_metrics.optimizer_step_seconds;
    metrics.obs_build_seconds = collector_timings.obs_build_seconds;
    metrics.mask_build_seconds = collector_timings.mask_build_seconds;
    metrics.env_step_seconds = collector_timings.env_step_seconds;
    metrics.reward_done_seconds = collector_timings.reward_done_seconds;

    metrics.collection_agent_steps_per_second =
        collected_agent_steps > 0 ? static_cast<double>(collected_agent_steps) / collection_seconds : 0.0;
    metrics.update_agent_steps_per_second =
        collected_agent_steps > 0 ? static_cast<double>(collected_agent_steps) / std::max(metrics.update_seconds, 1.0e-9) : 0.0;

    global_step += collected_agent_steps;

    if (self_play_manager_) {
      SelfPlayMetrics self_play_metrics =
          self_play_manager_->on_update(model_, normalizer_, global_step, update_index + 1);
      metrics.self_play_eval_seconds = self_play_metrics.eval_seconds;
      metrics.elo_ratings = std::move(self_play_metrics.ratings);
    }

    metrics.overall_agent_steps_per_second =
        collected_agent_steps > 0
            ? static_cast<double>(collected_agent_steps) /
                  std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count(), 1.0e-9)
            : 0.0;
    const torch::Tensor active_rewards = rollout_.rewards * rollout_.learner_active;
    const double active_count = rollout_.learner_active.sum().item<double>();
    metrics.reward_mean = active_count > 0.0 ? active_rewards.sum().item<double>() / active_count : 0.0;

    std::cout << "update=" << update_index
              << " collection_sps=" << metrics.collection_agent_steps_per_second
              << " update_sps=" << metrics.update_agent_steps_per_second
              << " overall_sps=" << metrics.overall_agent_steps_per_second
              << " reward_mean=" << metrics.reward_mean
              << " policy_loss=" << metrics.policy_loss
              << " value_loss=" << metrics.value_loss
              << " entropy=" << metrics.entropy
              << " self_play_eval_s=" << metrics.self_play_eval_seconds
              << '\n';
    append_metrics_line(checkpoint_dir, update_index, global_step, metrics);
    if (wandb.enabled()) {
      nlohmann::json wandb_metrics = {
          {"update", update_index},
          {"global_step", global_step},
          {"collection_agent_steps_per_second", metrics.collection_agent_steps_per_second},
          {"update_agent_steps_per_second", metrics.update_agent_steps_per_second},
          {"overall_agent_steps_per_second", metrics.overall_agent_steps_per_second},
          {"update_seconds", metrics.update_seconds},
          {"reward_mean", metrics.reward_mean},
          {"policy_loss", metrics.policy_loss},
          {"value_loss", metrics.value_loss},
          {"entropy", metrics.entropy},
          {"value_entropy", metrics.value_entropy},
          {"value_variance", metrics.value_variance},
          {"obs_build_seconds", metrics.obs_build_seconds},
          {"mask_build_seconds", metrics.mask_build_seconds},
          {"policy_forward_seconds", metrics.policy_forward_seconds},
          {"action_decode_seconds", metrics.action_decode_seconds},
          {"env_step_seconds", metrics.env_step_seconds},
          {"reward_done_seconds", metrics.reward_done_seconds},
          {"ngp_reward_seconds", metrics.ngp_reward_seconds},
          {"rollout_append_seconds", metrics.rollout_append_seconds},
          {"gae_seconds", metrics.gae_seconds},
          {"ppo_forward_backward_seconds", metrics.ppo_forward_backward_seconds},
          {"optimizer_step_seconds", metrics.optimizer_step_seconds},
          {"self_play_eval_seconds", metrics.self_play_eval_seconds},
      };
      for (const auto& [mode, rating] : metrics.elo_ratings) {
        wandb_metrics["elo/" + mode] = rating;
      }
      wandb.log(std::move(wandb_metrics));
    }

    if ((update_index + 1) % config_.ppo.checkpoint_interval == 0) {
      save_checkpoint(checkpoint_dir, global_step, update_index + 1);
    }
    if (metrics.reward_mean > best_reward_mean_) {
      best_reward_mean_ = metrics.reward_mean;
      save_checkpoint_to_directory(std::filesystem::path(checkpoint_dir) / "best", global_step, update_index + 1);
    }
  }
  save_checkpoint_to_directory(std::filesystem::path(checkpoint_dir) / "final", global_step, updates);
  wandb.finish();
}

}  // namespace pulsar

#endif
