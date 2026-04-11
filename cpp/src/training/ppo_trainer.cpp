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
  };

  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
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
      model_(SharedActorCritic(config_.model)),
      normalizer_(config_.model.observation_dim),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate)),
      rollout_(
          config_.ppo.rollout_length,
          static_cast<int>(total_agents_for(engines_)),
          config_.model.observation_dim,
          config_.model.action_dim,
          torch::Device(config_.ppo.device)),
      collection_executor_(static_cast<std::size_t>(config_.ppo.collection_workers)),
      device_(config_.ppo.device) {
  if (engines_.empty()) {
    throw std::invalid_argument("PPOTrainer requires at least one transition engine.");
  }

  total_agents_ = total_agents_for(engines_);
  agent_offsets_ = offsets_for(engines_);
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
  host_rewards_ = torch::empty(
      {static_cast<long>(total_agents_)},
      host_options);
  host_dones_ = torch::empty(
      {static_cast<long>(total_agents_)},
      host_options);
  model_->to(device_);
  normalizer_.to(device_);
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
  auto action_cpu = actions.to(torch::kCPU);
  std::memcpy(action_indices.data(), action_cpu.data_ptr<std::int64_t>(), action_indices.size() * sizeof(std::int64_t));
  return action_indices;
}

TrainerMetrics PPOTrainer::update_policy() {
  const auto update_start = std::chrono::steady_clock::now();

  torch::Tensor flat_obs = rollout_.obs.reshape({-1, config_.model.observation_dim});
  torch::Tensor flat_actions = rollout_.actions.reshape({-1});
  torch::Tensor flat_old_log_probs = rollout_.log_probs.reshape({-1});
  torch::Tensor flat_returns = rollout_.returns.reshape({-1});
  torch::Tensor flat_advantages = rollout_.advantages.reshape({-1});

  flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-6);

  const auto batch_size = flat_obs.size(0);
  TrainerMetrics metrics{};

  for (int epoch = 0; epoch < config_.ppo.epochs; ++epoch) {
    const torch::Tensor indices = torch::randperm(batch_size, torch::TensorOptions().device(device_).dtype(torch::kLong));

    for (int offset = 0; offset < batch_size; offset += config_.ppo.minibatch_size) {
      const int length = std::min(config_.ppo.minibatch_size, static_cast<int>(batch_size - offset));
      const torch::Tensor batch_indices = indices.narrow(0, offset, length);

      const torch::Tensor obs = flat_obs.index_select(0, batch_indices);
      const torch::Tensor actions = flat_actions.index_select(0, batch_indices);
      const torch::Tensor old_log_probs = flat_old_log_probs.index_select(0, batch_indices);
      const torch::Tensor returns = flat_returns.index_select(0, batch_indices);
      const torch::Tensor advantages = flat_advantages.index_select(0, batch_indices);

      const PolicyOutput output = model_->forward(obs);
      const torch::Tensor log_probs =
          torch::log_softmax(output.logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
      const torch::Tensor entropy =
          -(torch::softmax(output.logits, -1) * torch::log_softmax(output.logits, -1)).sum(-1).mean();
      const torch::Tensor ratio = torch::exp(log_probs - old_log_probs);
      const torch::Tensor clipped_ratio =
          torch::clamp(ratio, 1.0 - config_.ppo.clip_range, 1.0 + config_.ppo.clip_range);
      const torch::Tensor policy_loss = -torch::min(ratio * advantages, clipped_ratio * advantages).mean();
      const torch::Tensor value_loss = torch::mse_loss(output.values, returns);
      const torch::Tensor loss =
          policy_loss + config_.ppo.value_coef * value_loss - config_.ppo.entropy_coef * entropy;

      optimizer_.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.ppo.max_grad_norm);
      optimizer_.step();

      metrics.policy_loss = policy_loss.item<double>();
      metrics.value_loss = value_loss.item<double>();
      metrics.entropy = entropy.item<double>();
    }
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
      .architecture_name = "shared_actor_critic",
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

void PPOTrainer::train(int updates, const std::string& checkpoint_dir) {
  std::int64_t global_step = 0;

  for (int update_index = 0; update_index < updates; ++update_index) {
    const auto collection_start = std::chrono::steady_clock::now();

    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      torch::Tensor obs = collect_observations();
      torch::Tensor actions;
      torch::Tensor log_probs;
      torch::Tensor values;
      {
        torch::NoGradGuard no_grad;
        normalizer_.update(obs);
        obs = normalizer_.normalize(obs);

        const PolicyOutput output = model_->forward(obs);
        actions = sample_actions(output.logits, &log_probs);
        values = output.values;
      }
      const std::vector<std::int64_t> action_indices = actions_to_indices(actions);
      step_envs(action_indices, global_step);

      rollout_.append(
          step,
          obs,
          actions,
          log_probs,
          host_rewards_.to(device_, use_pinned_host_buffers_),
          host_dones_.to(device_, use_pinned_host_buffers_),
          values);

      global_step += static_cast<std::int64_t>(total_agents_);
    }

    torch::Tensor last_value;
    {
      torch::NoGradGuard no_grad;
      torch::Tensor last_obs = normalizer_.normalize(collect_observations());
      last_value = model_->forward(last_obs).values;
    }
    rollout_.compute_returns_and_advantages(last_value, config_.ppo.gamma, config_.ppo.gae_lambda);

    TrainerMetrics metrics = update_policy();
    metrics.collection_fps =
        static_cast<double>(config_.ppo.rollout_length * total_agents_) /
        std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

    std::cout << "update=" << update_index
              << " fps=" << metrics.collection_fps
              << " policy_loss=" << metrics.policy_loss
              << " value_loss=" << metrics.value_loss
              << " entropy=" << metrics.entropy
              << '\n';
    append_metrics_line(checkpoint_dir, update_index, global_step, metrics);

    if ((update_index + 1) % config_.ppo.checkpoint_interval == 0) {
      save_checkpoint(checkpoint_dir, global_step, update_index + 1);
    }
  }
}

}  // namespace pulsar

#endif
