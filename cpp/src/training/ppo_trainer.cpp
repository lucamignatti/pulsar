#include "pulsar/training/ppo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pulsar {
namespace {

std::size_t total_agents_for(const std::vector<TransitionEnginePtr>& engines) {
  std::size_t total = 0;
  for (const auto& engine : engines) {
    total += engine->num_agents();
  }
  return total;
}

std::vector<std::size_t> offsets_for(const std::vector<TransitionEnginePtr>& engines) {
  std::vector<std::size_t> offsets;
  offsets.reserve(engines.size() + 1);
  offsets.push_back(0);
  std::size_t total = 0;
  for (const auto& engine : engines) {
    total += engine->num_agents();
    offsets.push_back(total);
  }
  return offsets;
}

void append_metrics_line(
    const std::filesystem::path& checkpoint_dir,
    int update_index,
    std::int64_t global_step,
    const TrainerMetrics& metrics) {
  nlohmann::json line = {
      {"update", update_index},
      {"global_step", global_step},
      {"collection_fps", metrics.collection_fps},
      {"update_seconds", metrics.update_seconds},
      {"policy_loss", metrics.policy_loss},
      {"value_loss", metrics.value_loss},
      {"entropy", metrics.entropy},
      {"value_entropy", metrics.value_entropy},
      {"value_variance", metrics.value_variance},
  };

  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

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

}  // namespace

PPOTrainer::PPOTrainer(
    ExperimentConfig config,
    std::vector<TransitionEnginePtr> engines,
    ObsBuilderPtr obs_builder,
    ActionParserPtr action_parser,
    RewardFunctionPtr reward_fn,
    DoneConditionPtr done_condition)
    : config_(std::move(config)),
      engines_(std::move(engines)),
      obs_builder_(std::move(obs_builder)),
      action_parser_(std::move(action_parser)),
      reward_fn_(std::move(reward_fn)),
      done_condition_(std::move(done_condition)),
      action_table_(config_.action_table),
      model_(SharedActorCritic(config_.model, config_.ppo)),
      normalizer_(config_.model.observation_dim),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate)),
      rollout_(
          config_.ppo.rollout_length,
          static_cast<int>(total_agents_for(engines_)),
          config_.model.observation_dim,
          torch::Device(config_.ppo.device)),
      device_(config_.ppo.device),
      collection_executor_(static_cast<std::size_t>(config_.ppo.collection_workers)) {
  if (engines_.empty()) {
    throw std::invalid_argument("PPOTrainer requires at least one transition engine.");
  }

  total_agents_ = total_agents_for(engines_);
  agent_offsets_ = offsets_for(engines_);
  collection_state_ = model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  host_actions_.resize(total_agents_);
  host_terminated_.resize(total_agents_, 0);
  host_truncated_.resize(total_agents_, 0);
  use_pinned_host_buffers_ = device_.is_cuda();
  auto host_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  if (use_pinned_host_buffers_) {
    host_options = host_options.pinned_memory(true);
  }
  host_obs_ = torch::empty(
      {static_cast<long>(total_agents_), static_cast<long>(config_.model.observation_dim)},
      host_options);
  host_episode_starts_ = torch::ones({static_cast<long>(total_agents_)}, host_options);
  host_rewards_ = torch::empty({static_cast<long>(total_agents_)}, host_options);
  host_dones_ = torch::empty({static_cast<long>(total_agents_)}, host_options);
  model_->to(device_);
  normalizer_.to(device_);
  maybe_initialize_from_checkpoint();
}

torch::Tensor PPOTrainer::collect_observations() {
  float* dst = host_obs_.data_ptr<float>();
  const std::size_t stride = obs_builder_->obs_dim();
  collection_executor_.parallel_for(engines_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t engine_idx = begin; engine_idx < end; ++engine_idx) {
      const auto& engine = engines_[engine_idx];
      const std::size_t agent_offset = agent_offsets_[engine_idx];
      const std::size_t count = engine->num_agents();
      obs_builder_->build_obs_batch(
          engine->state(),
          std::span<float>(
              dst + static_cast<std::ptrdiff_t>(agent_offset * stride),
              count * stride));
    }
  });
  return host_obs_.to(device_, use_pinned_host_buffers_);
}

void PPOTrainer::step_envs(std::span<const std::int64_t> action_indices, std::int64_t global_step) {
  float* rewards_ptr = host_rewards_.data_ptr<float>();
  float* dones_ptr = host_dones_.data_ptr<float>();

  collection_executor_.parallel_for(engines_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t engine_idx = begin; engine_idx < end; ++engine_idx) {
      const std::size_t agent_begin = agent_offsets_[engine_idx];
      const std::size_t agent_end = agent_offsets_[engine_idx + 1];
      const std::size_t agent_count = agent_end - agent_begin;

      const std::span<const std::int64_t> env_action_indices(
          action_indices.data() + static_cast<std::ptrdiff_t>(agent_begin),
          agent_count);
      const std::span<ControllerState> env_actions(
          host_actions_.data() + static_cast<std::ptrdiff_t>(agent_begin),
          agent_count);
      action_parser_->parse_actions_into(env_action_indices, env_actions);

      const EnvState previous_state = engines_[engine_idx]->state();
      engines_[engine_idx]->step_inplace(env_actions);
      const EnvState& current_state = engines_[engine_idx]->state();

      const std::span<std::uint8_t> terminated(
          host_terminated_.data() + static_cast<std::ptrdiff_t>(agent_begin),
          agent_count);
      const std::span<std::uint8_t> truncated(
          host_truncated_.data() + static_cast<std::ptrdiff_t>(agent_begin),
          agent_count);
      done_condition_->is_done_into(current_state, current_state.tick, terminated, truncated);
      reward_fn_->get_rewards_into(
          previous_state,
          current_state,
          terminated,
          truncated,
          std::span<float>(rewards_ptr + static_cast<std::ptrdiff_t>(agent_begin), agent_count));

      bool reset_needed = false;
      for (std::size_t i = 0; i < agent_count; ++i) {
        const bool done = terminated[i] != 0 || truncated[i] != 0;
        dones_ptr[agent_begin + i] = done ? 1.0F : 0.0F;
        reset_needed = reset_needed || done;
      }

      if (reset_needed) {
        engines_[engine_idx]->reset(
            config_.env.seed + static_cast<std::uint64_t>(global_step) + static_cast<std::uint64_t>(engine_idx));
      }
    }
  });
}

ContinuumState PPOTrainer::replay_state_until(std::int64_t start_step, const torch::Tensor& agent_indices) {
  ContinuumState state = model_->initial_state(agent_indices.size(0), device_);
  if (start_step <= 0) {
    return state;
  }

  torch::NoGradGuard no_grad;
  for (std::int64_t step = 0; step < start_step; ++step) {
    const torch::Tensor obs = rollout_.obs[step].index_select(0, agent_indices);
    const torch::Tensor starts = rollout_.episode_starts[step].index_select(0, agent_indices);
    SequenceOutput out = model_->forward_sequence(obs.unsqueeze(0), std::move(state), starts.unsqueeze(0));
    state = std::move(out.final_state);
  }
  return state;
}

torch::Tensor PPOTrainer::sample_actions(const torch::Tensor& logits, torch::Tensor* log_probs) const {
  const torch::Tensor probs = torch::softmax(logits, -1);
  const torch::Tensor actions = probs.multinomial(1).squeeze(-1);
  const torch::Tensor chosen_log_probs =
      torch::log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
  if (log_probs != nullptr) {
    *log_probs = chosen_log_probs;
  }
  return actions;
}

std::vector<std::int64_t> PPOTrainer::actions_to_indices(const torch::Tensor& actions) const {
  std::vector<std::int64_t> action_indices(static_cast<std::size_t>(actions.size(0)));
  const torch::Tensor action_cpu = actions.to(torch::kCPU);
  std::memcpy(action_indices.data(), action_cpu.data_ptr<std::int64_t>(), action_indices.size() * sizeof(std::int64_t));
  return action_indices;
}

torch::Tensor PPOTrainer::categorical_projection(const torch::Tensor& returns) const {
  const torch::Tensor support = model_->support().to(returns.device());
  const float v_min = config_.ppo.value_v_min;
  const float v_max = config_.ppo.value_v_max;
  const float delta_z = (v_max - v_min) / static_cast<float>(config_.ppo.value_num_atoms - 1);
  const torch::Tensor clamped = returns.clamp(v_min, v_max);
  const torch::Tensor b = (clamped - v_min) / delta_z;
  const torch::Tensor lower = b.floor().to(torch::kLong).clamp(0, config_.ppo.value_num_atoms - 1);
  const torch::Tensor upper = b.ceil().to(torch::kLong).clamp(0, config_.ppo.value_num_atoms - 1);
  const torch::Tensor upper_prob = b - lower.to(torch::kFloat32);
  const torch::Tensor lower_prob = 1.0 - upper_prob;
  torch::Tensor target = torch::zeros(
      {returns.size(0), config_.ppo.value_num_atoms},
      torch::TensorOptions().dtype(torch::kFloat32).device(returns.device()));
  target.scatter_add_(1, lower.unsqueeze(-1), lower_prob.unsqueeze(-1));
  target.scatter_add_(1, upper.unsqueeze(-1), upper_prob.unsqueeze(-1));
  (void)support;
  return target;
}

torch::Tensor PPOTrainer::confidence_weights(const torch::Tensor& value_logits) const {
  if (!config_.ppo.use_confidence_weighting) {
    return torch::ones({value_logits.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(value_logits.device()));
  }
  torch::Tensor weights;
  if (config_.ppo.confidence_weight_type == "variance") {
    weights = 1.0 / (model_->value_variance(value_logits) + config_.ppo.confidence_weight_delta);
  } else {
    weights = 1.0 / (model_->value_entropy(value_logits) + config_.ppo.confidence_weight_delta);
  }
  if (config_.ppo.normalize_confidence_weights) {
    weights = weights / weights.mean().clamp_min(1.0e-6);
  }
  return weights.detach();
}

torch::Tensor PPOTrainer::adaptive_epsilon(const torch::Tensor& value_logits) const {
  if (!config_.ppo.use_adaptive_epsilon) {
    return torch::full(
        {value_logits.size(0)},
        config_.ppo.clip_range,
        torch::TensorOptions().dtype(torch::kFloat32).device(value_logits.device()));
  }
  torch::Tensor epsilon = config_.ppo.clip_range /
      (1.0 + config_.ppo.adaptive_epsilon_beta * model_->value_variance(value_logits));
  return torch::clamp(epsilon, config_.ppo.epsilon_min, config_.ppo.epsilon_max).detach();
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

TrainerMetrics PPOTrainer::update_policy() {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
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
        const torch::Tensor actions =
            rollout_.actions.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor old_log_probs =
            rollout_.log_probs.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor returns =
            rollout_.returns.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        torch::Tensor advantages =
            rollout_.advantages.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-6);
        const SequenceOutput output = model_->forward_sequence(obs, std::move(state), episode_starts);
        torch::Tensor policy_logits = output.policy_logits.narrow(0, burn, effective_end - effective_start);
        torch::Tensor value_logits = output.value_logits.narrow(0, burn, effective_end - effective_start);

        const torch::Tensor flat_policy_logits = policy_logits.reshape({-1, config_.model.action_dim});
        const torch::Tensor flat_value_logits = value_logits.reshape({-1, config_.ppo.value_num_atoms});
        const torch::Tensor flat_actions = actions.reshape({-1});
        const torch::Tensor flat_old_log_probs = old_log_probs.reshape({-1});
        const torch::Tensor flat_advantages = advantages.reshape({-1});
        const torch::Tensor flat_returns = returns.reshape({-1});

        const torch::Tensor current_log_probs =
            torch::log_softmax(flat_policy_logits, -1).gather(-1, flat_actions.unsqueeze(-1)).squeeze(-1);
        const torch::Tensor ratio = torch::exp(current_log_probs - flat_old_log_probs);
        const torch::Tensor epsilon = adaptive_epsilon(flat_value_logits);
        const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - epsilon, 1.0 + epsilon);
        const torch::Tensor weights = confidence_weights(flat_value_logits);
        const torch::Tensor unclipped = ratio * flat_advantages;
        const torch::Tensor clipped = clipped_ratio * flat_advantages;
        const torch::Tensor policy_loss = -(torch::min(unclipped, clipped) * weights).mean();

        const torch::Tensor action_entropy =
            -(torch::softmax(flat_policy_logits, -1) * torch::log_softmax(flat_policy_logits, -1)).sum(-1).mean();
        const torch::Tensor target_dist = categorical_projection(flat_returns);
        const torch::Tensor critic_loss =
            -(target_dist * torch::log_softmax(flat_value_logits, -1)).sum(-1).mean();
        const torch::Tensor loss =
            policy_loss + config_.ppo.value_coef * critic_loss - config_.ppo.entropy_coef * action_entropy;

        optimizer_.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.ppo.max_grad_norm);
        optimizer_.step();

        metrics.policy_loss += policy_loss.item<double>() * flat_actions.size(0);
        metrics.value_loss += critic_loss.item<double>() * flat_actions.size(0);
        metrics.entropy += action_entropy.item<double>() * flat_actions.size(0);
        metrics.value_entropy += model_->value_entropy(flat_value_logits).mean().item<double>() * flat_actions.size(0);
        metrics.value_variance += model_->value_variance(flat_value_logits).mean().item<double>() * flat_actions.size(0);
        metric_steps += flat_actions.size(0);
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

  for (int update_index = 0; update_index < updates; ++update_index) {
    const auto collection_start = std::chrono::steady_clock::now();

    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      torch::Tensor obs = collect_observations();
      torch::Tensor actions;
      torch::Tensor log_probs;
      torch::Tensor sampled_values;
      torch::Tensor episode_starts = host_episode_starts_.to(device_, use_pinned_host_buffers_);
      {
        torch::NoGradGuard no_grad;
        normalizer_.update(obs);
        obs = normalizer_.normalize(obs);

        PolicyOutput output = model_->forward_step(obs, std::move(collection_state_), episode_starts);
        collection_state_ = std::move(output.state);
        actions = sample_actions(output.policy_logits, &log_probs);
        sampled_values = output.sampled_values;
      }

      const std::vector<std::int64_t> action_indices = actions_to_indices(actions);
      step_envs(action_indices, global_step);
      const torch::Tensor rewards = host_rewards_.to(device_, use_pinned_host_buffers_);
      const torch::Tensor dones = host_dones_.to(device_, use_pinned_host_buffers_);

      rollout_.append(step, obs, episode_starts, actions, log_probs, rewards, dones, sampled_values);
      host_episode_starts_.copy_(host_dones_);
      global_step += static_cast<std::int64_t>(total_agents_);
    }

    torch::Tensor last_sampled_value;
    {
      torch::NoGradGuard no_grad;
      torch::Tensor last_obs = normalizer_.normalize(collect_observations());
      torch::Tensor episode_starts = host_episode_starts_.to(device_, use_pinned_host_buffers_);
      PolicyOutput output = model_->forward_step(last_obs, collection_state_, episode_starts);
      last_sampled_value = output.sampled_values;
    }
    rollout_.compute_returns_and_advantages(last_sampled_value, config_.ppo.gamma, config_.ppo.gae_lambda);

    TrainerMetrics metrics = update_policy();
    metrics.collection_fps =
        static_cast<double>(config_.ppo.rollout_length * total_agents_) /
        std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

    std::cout << "update=" << update_index
              << " fps=" << metrics.collection_fps
              << " policy_loss=" << metrics.policy_loss
              << " value_loss=" << metrics.value_loss
              << " entropy=" << metrics.entropy
              << " value_entropy=" << metrics.value_entropy
              << " value_variance=" << metrics.value_variance
              << '\n';
    append_metrics_line(checkpoint_dir, update_index, global_step, metrics);
    if (wandb.enabled()) {
      wandb.log(nlohmann::json{
          {"update", update_index},
          {"global_step", global_step},
          {"collection_fps", metrics.collection_fps},
          {"update_seconds", metrics.update_seconds},
          {"policy_loss", metrics.policy_loss},
          {"value_loss", metrics.value_loss},
          {"entropy", metrics.entropy},
          {"value_entropy", metrics.value_entropy},
          {"value_variance", metrics.value_variance},
      });
    }

    if ((update_index + 1) % config_.ppo.checkpoint_interval == 0) {
      save_checkpoint(checkpoint_dir, global_step, update_index + 1);
    }
  }
  wandb.finish();
}

}  // namespace pulsar

#endif
