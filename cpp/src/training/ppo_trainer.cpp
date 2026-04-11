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
      device_(config_.ppo.device) {
  if (engines_.empty()) {
    throw std::invalid_argument("PPOTrainer requires at least one transition engine.");
  }

  total_agents_ = total_agents_for(engines_);
  agent_offsets_ = offsets_for(engines_);
  model_->to(device_);
}

torch::Tensor PPOTrainer::collect_observations() const {
  std::vector<float> flat;
  flat.reserve(total_agents_ * obs_builder_->obs_dim());

  for (const auto& engine : engines_) {
    for (std::size_t agent_id = 0; agent_id < engine->num_agents(); ++agent_id) {
      const auto obs = obs_builder_->build_obs(engine->state(), agent_id);
      flat.insert(flat.end(), obs.begin(), obs.end());
    }
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
  torch::Tensor host = torch::from_blob(
      flat.data(),
      {static_cast<long>(total_agents_), static_cast<long>(obs_builder_->obs_dim())},
      options)
                           .clone();
  return host.to(device_, true);
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
      std::vector<EnvState> previous_states;
      previous_states.reserve(engines_.size());
      for (const auto& engine : engines_) {
        previous_states.push_back(engine->state());
      }

      torch::Tensor obs = collect_observations();
      normalizer_.update(obs);
      obs = normalizer_.normalize(obs);

      const PolicyOutput output = model_->forward(obs);
      torch::Tensor log_probs;
      const torch::Tensor actions = sample_actions(output.logits, &log_probs);
      const std::vector<std::int64_t> action_indices = actions_to_indices(actions);

      std::vector<float> rewards_flat;
      std::vector<float> dones_flat;
      rewards_flat.reserve(total_agents_);
      dones_flat.reserve(total_agents_);

      for (std::size_t engine_idx = 0; engine_idx < engines_.size(); ++engine_idx) {
        const std::size_t begin = agent_offsets_[engine_idx];
        const std::size_t end = agent_offsets_[engine_idx + 1];
        std::vector<std::int64_t> env_action_indices(
            action_indices.begin() + static_cast<std::ptrdiff_t>(begin),
            action_indices.begin() + static_cast<std::ptrdiff_t>(end));
        const std::vector<ControllerState> parsed_actions = action_parser_->parse_actions(env_action_indices);
        const StepResult transition = engines_[engine_idx]->step(parsed_actions);
        const auto [terminated, truncated] = done_condition_->is_done(transition.state, transition.state.tick);
        const std::vector<float> rewards =
            reward_fn_->get_rewards(previous_states[engine_idx], transition.state, terminated, truncated);

        for (std::size_t i = 0; i < rewards.size(); ++i) {
          rewards_flat.push_back(rewards[i]);
          dones_flat.push_back((terminated[i] != 0 || truncated[i] != 0) ? 1.0F : 0.0F);
        }

        const bool reset_needed =
            std::any_of(terminated.begin(), terminated.end(), [](std::uint8_t value) { return value != 0; }) ||
            std::any_of(truncated.begin(), truncated.end(), [](std::uint8_t value) { return value != 0; });
        if (reset_needed) {
          engines_[engine_idx]->reset(
              config_.env.seed + static_cast<std::uint64_t>(global_step) + static_cast<std::uint64_t>(engine_idx));
        }
      }

      rollout_.append(
          step,
          obs,
          actions,
          log_probs,
          torch::tensor(rewards_flat, torch::TensorOptions().dtype(torch::kFloat32).device(device_)),
          torch::tensor(dones_flat, torch::TensorOptions().dtype(torch::kFloat32).device(device_)),
          output.values.detach());

      global_step += static_cast<std::int64_t>(total_agents_);
    }

    torch::Tensor last_obs = normalizer_.normalize(collect_observations());
    torch::Tensor last_value = model_->forward(last_obs).values.detach();
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
