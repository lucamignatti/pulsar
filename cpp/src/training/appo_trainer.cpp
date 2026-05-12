#include "pulsar/training/appo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/training/cuda_utils.hpp"
#include "pulsar/training/curriculum.hpp"
#include "pulsar/training/ppo_math.hpp"
#include "pulsar/tracing/tracing.hpp"

namespace pulsar {
namespace {

torch::Device resolve_runtime_device(const std::string& device_name) {
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

void synchronize_cuda_if_needed(const torch::Device& device, const char* context) noexcept {
  if (!device.is_cuda()) {
    return;
  }
  try {
    torch::cuda::synchronize();
  } catch (const std::exception& exc) {
    std::cerr << "cuda synchronize failed during " << context << ": " << exc.what() << '\n';
  }
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

void require_finite(const torch::Tensor& tensor, const std::string& name) {
  if (tensor.defined() && !torch::isfinite(tensor).all().item<bool>()) {
    throw std::runtime_error("Non-finite tensor: " + name);
  }
}

void shrink_perturb_parameters(torch::nn::Module& module, float shrink, float noise) {
  torch::NoGradGuard no_grad;
  for (auto& param : module.named_parameters()) {
    if (!param.value().requires_grad() || param.value().dim() < 2) {
      continue;
    }
    const std::string& name = param.key();
    if (name.find("lora_") != std::string::npos) {
      continue;
    }
    if (name == "pos" || name.rfind(".pos") == name.size() - 4) {
      continue;
    }
    param.value().mul_(shrink);
    param.value().add_(torch::randn_like(param.value()) * noise);
  }
}

struct CapturedGrad {
  torch::Tensor param;
  torch::Tensor grad;
};

std::vector<CapturedGrad> capture_gradients(torch::nn::Module& module) {
  std::vector<CapturedGrad> out;
  for (auto& p : module.parameters()) {
    if (p.grad().defined()) {
      out.push_back({p, p.grad().detach().clone()});
    } else {
      out.push_back({p, torch::Tensor{}});
    }
  }
  return out;
}

void zero_existing_gradients(torch::nn::Module& module) {
  for (auto& p : module.parameters()) {
    torch::Tensor grad = p.mutable_grad();
    if (grad.defined()) {
      grad.zero_();
    }
  }
}

void apply_pcgrad(std::vector<CapturedGrad>& group_a, std::vector<CapturedGrad>& group_b) {
  std::vector<torch::Tensor> flat_a_parts, flat_b_parts;
  for (size_t i = 0; i < group_a.size(); ++i) {
    if (group_a[i].grad.defined() && group_b[i].grad.defined()) {
      flat_a_parts.push_back(group_a[i].grad.view({-1}));
      flat_b_parts.push_back(group_b[i].grad.view({-1}));
    }
  }
  if (flat_a_parts.empty()) return;

  torch::Tensor ga_all = torch::cat(flat_a_parts, 0);
  torch::Tensor gb_all = torch::cat(flat_b_parts, 0);

  float dot = ga_all.dot(gb_all).item<float>();
  if (dot < 0.0F) {
    float norm_a_sq = ga_all.dot(ga_all).item<float>() + 1.0e-12F;
    float norm_b_sq = gb_all.dot(gb_all).item<float>() + 1.0e-12F;
    torch::Tensor ga_orig = ga_all.clone();
    torch::Tensor gb_orig = gb_all.clone();
    ga_all = ga_orig - (dot / norm_b_sq) * gb_orig;
    gb_all = gb_orig - (dot / norm_a_sq) * ga_orig;

    size_t offset_a = 0, offset_b = 0;
    for (size_t i = 0; i < group_a.size(); ++i) {
      if (group_a[i].grad.defined() && group_b[i].grad.defined()) {
        auto sz = static_cast<int64_t>(group_a[i].grad.numel());
        group_a[i].grad = ga_all.slice(0, static_cast<int64_t>(offset_a), static_cast<int64_t>(offset_a) + sz).view(group_a[i].grad.sizes()).clone();
        group_b[i].grad = gb_all.slice(0, static_cast<int64_t>(offset_b), static_cast<int64_t>(offset_b) + sz).view(group_b[i].grad.sizes()).clone();
        offset_a += static_cast<size_t>(sz);
        offset_b += static_cast<size_t>(sz);
      }
    }
  }
}

torch::Tensor policy_goal_values_like(const torch::Tensor& obs, int goal_dim) {
  const auto options = obs.options().dtype(torch::kFloat32);
  if (obs.dim() == 3) {
    return torch::zeros({obs.size(0), obs.size(1), goal_dim}, options);
  }
  return torch::zeros({obs.size(0), goal_dim}, options);
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
      {"grad_norm", metrics.grad_norm},
      {"total_reward_mean", metrics.total_reward_mean},
      {"gameplay_reward_mean", metrics.gameplay_reward_mean},
      {"mechanic_reward_mean", metrics.mechanic_reward_mean},
      {"sampled_value_win_mean", metrics.sampled_value_win_mean},
      {"rollout_steps", metrics.rollout_steps},
      {"completed_episodes", metrics.completed_episodes},
      {"scored_episodes", metrics.scored_episodes},
      {"goal_critic_loss", metrics.goal_critic_loss},
      {"mean_goal_score", metrics.mean_goal_score},
      {"mean_sampled_goal_distance", metrics.mean_sampled_goal_distance},
      {"mean_goal_distance", metrics.mean_goal_distance},
      {"min_goal_distance", metrics.min_goal_distance},
      {"ball_proximity_rate", metrics.ball_proximity_rate},
      {"goals_scored", metrics.goals_scored},
      {"goals_conceded", metrics.goals_conceded},
      {"obs_build_seconds", metrics.obs_build_seconds},
      {"mask_build_seconds", metrics.mask_build_seconds},
      {"policy_forward_seconds", metrics.policy_forward_seconds},
      {"action_decode_seconds", metrics.action_decode_seconds},
      {"env_step_seconds", metrics.env_step_seconds},
      {"done_reset_seconds", metrics.done_reset_seconds},
      {"forward_backward_seconds", metrics.forward_backward_seconds},
      {"optimizer_step_seconds", metrics.optimizer_step_seconds},
      {"self_play_eval_seconds", metrics.self_play_eval_seconds},
      {"es_fitness_mean", metrics.es_fitness_mean},
      {"es_fitness_std", metrics.es_fitness_std},
      {"es_fitness_best", metrics.es_fitness_best},
      {"es_winrate_mean", metrics.es_winrate_mean},
      {"es_kl_mean", metrics.es_kl_mean},
      {"es_update_norm", metrics.es_update_norm},
      {"es_lora_a_norm", metrics.es_lora_a_norm},
      {"es_lora_b_norm", metrics.es_lora_b_norm},
      {"es_seconds", metrics.es_seconds},
      {"scored_episode_rate", metrics.scored_episode_rate},
      {"effective_entropy_coef", metrics.effective_entropy_coef},
  };
  for (const auto& [mode, rating] : metrics.elo_ratings) {
    line["elo_" + mode] = rating;
  }
  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

std::shared_ptr<MutatorSequence> make_es_eval_reset_mutator(const EnvConfig& config) {
  return std::make_shared<MutatorSequence>(
      std::vector<StateMutatorPtr>{
          std::make_shared<FixedTeamSizeMutator>(config),
          std::make_shared<KickoffMutator>(config),
      });
}

std::unique_ptr<BatchedRocketSimCollector> make_es_eval_collector(
    const ExperimentConfig& config,
    int total_envs,
    int eval_envs_per_member,
    int update_index,
    int episode_index,
    bool pin_host_memory) {
  ExperimentConfig eval_config = config;
  eval_config.ppo.num_envs = total_envs;
  eval_config.ppo.collection_workers = std::min(config.ppo.collection_workers, total_envs);

  const auto reset_mutator = make_es_eval_reset_mutator(config.env);
  std::vector<TransitionEnginePtr> engines;
  engines.reserve(static_cast<std::size_t>(total_envs));
  for (int env_idx = 0; env_idx < total_envs; ++env_idx) {
    const int local_env = env_idx % eval_envs_per_member;
    EnvConfig env_config = config.env;
    env_config.seed += static_cast<std::uint64_t>(
        1'000'003 + update_index * 65'537 + episode_index * 8'191 + local_env);
    engines.push_back(std::make_shared<RocketSimTransitionEngine>(env_config, reset_mutator));
  }

  return std::make_unique<BatchedRocketSimCollector>(
      eval_config,
      std::move(engines),
      std::make_shared<PulsarObsBuilder>(config.env),
      std::make_shared<DiscreteActionParser>(ControllerActionTable(config.action_table)),
      std::make_shared<SimpleDoneCondition>(config.env),
      pin_host_memory);
}

std::vector<std::unique_ptr<BatchedRocketSimCollector>> make_collector_vector(
    std::unique_ptr<BatchedRocketSimCollector> collector) {
  std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors;
  collectors.push_back(std::move(collector));
  return collectors;
}

std::size_t total_agents_for_collectors(
    const std::vector<std::unique_ptr<BatchedRocketSimCollector>>& collectors) {
  std::size_t total = 0;
  for (const auto& collector : collectors) {
    if (collector) {
      total += collector->total_agents();
    }
  }
  return total;
}

int action_dim_for_collectors(
    const std::vector<std::unique_ptr<BatchedRocketSimCollector>>& collectors) {
  for (const auto& collector : collectors) {
    if (collector) {
      return collector->action_dim();
    }
  }
  return 0;
}

void accumulate_timings(CollectorTimings& dst, const CollectorTimings& src) {
  dst.obs_build_seconds += src.obs_build_seconds;
  dst.mask_build_seconds += src.mask_build_seconds;
  dst.env_step_seconds += src.env_step_seconds;
  dst.done_reset_seconds += src.done_reset_seconds;
}

}  // namespace

APPOTrainer::APPOTrainer(
    ExperimentConfig config,
    std::unique_ptr<BatchedRocketSimCollector> collector,
    std::unique_ptr<SelfPlayManager> self_play_manager,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : APPOTrainer(
          std::move(config),
          make_collector_vector(std::move(collector)),
          std::move(self_play_manager),
          std::move(run_output_root),
          log_initialization) {}

APPOTrainer::APPOTrainer(
    ExperimentConfig config,
    std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors,
    std::unique_ptr<SelfPlayManager> self_play_manager,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : config_(std::move(config)),
      collectors_(std::move(collectors)),
      self_play_manager_(std::move(self_play_manager)),
      curriculum_(config_.curriculum),
      action_table_(config_.action_table),
      actor_(PPOActor(config_.model, config_.goal_critic, config_.es_lora)),
      actor_normalizer_(config_.model.observation_dim),
      actor_optimizer_(actor_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate).eps(1.0e-5F)),
      rollout_(make_rollout_storage(
          config_,
          static_cast<int>(total_agents_for_collectors(collectors_)),
          action_dim_for_collectors(collectors_))),
      rollout_B_(make_rollout_storage(
          config_,
          static_cast<int>(total_agents_for_collectors(collectors_)),
          action_dim_for_collectors(collectors_))),
      device_(resolve_runtime_device(config_.ppo.device)),
      run_output_root_(std::move(run_output_root)),
      log_initialization_(log_initialization) {
  validate_experiment_config(config_);
  if (collectors_.empty()) {
    throw std::invalid_argument("APPOTrainer requires at least one collector.");
  }
  if (actor_->policy_lora()->out_features() != action_dim_for_collectors(collectors_)) {
    throw std::invalid_argument("model.action_dim must match the action table size.");
  }
  if (config_.model.observation_dim != collectors_.front()->obs_dim()) {
    throw std::invalid_argument("model.observation_dim must match obs builder output.");
  }
  total_agents_ = total_agents_for_collectors(collectors_);
  if (total_agents_ == 0) {
    throw std::invalid_argument("APPOTrainer collectors must contain agents.");
  }
  seed_everything(config_.env.seed);
  configure_cuda_runtime(device_);
  use_pinned_host_buffers_ = device_.is_cuda();
  actor_->to(device_);
  actor_normalizer_.to(device_);

  maybe_initialize_from_checkpoint();
  actor_snapshot_ = clone_ppo_actor(actor_, device_);
  actor_snapshot_->eval();

  shard_agent_offsets_.clear();
  std::int64_t agent_offset = 0;
  for (const auto& collector : collectors_) {
    if (!collector) {
      throw std::invalid_argument("APPOTrainer collectors must be non-null.");
    }
    shard_agent_offsets_.push_back(agent_offset);
    const auto shard_agents = static_cast<std::int64_t>(collector->total_agents());
    agent_offset += shard_agents;
  }

  if (self_play_manager_ && self_play_manager_->enabled()) {
    std::size_t env_offset = 0;
    for (auto& collector : collectors_) {
      const std::size_t shard_env_offset = env_offset;
      collector->set_self_play_assignment_fn(
          [this, shard_env_offset](std::size_t env_idx, std::uint64_t seed) {
            return self_play_manager_->sample_assignment(shard_env_offset + env_idx, seed);
          });
      env_offset += collector->num_envs();
    }
  }
}

APPOTrainer::~APPOTrainer() {
  synchronize_cuda_if_needed(device_, "trainer shutdown");
}

void APPOTrainer::apply_curriculum_to_collectors() {
  if (!curriculum_.enabled()) return;
  curriculum_.initialize_stage();
  auto cfg = config_;
  cfg.outcome = curriculum_.outcome();
  cfg.mechanic_rewards = curriculum_.mechanic_rewards();
  cfg.dense_rewards = curriculum_.dense_rewards();
  for (auto& collector : collectors_) {
    if (collector) {
      collector->update_reward_config(cfg);
      collector->update_unlocked_mechanics(curriculum_.unlocked_mechanics());
    }
  }
  if (self_play_manager_) {
    self_play_manager_->set_curriculum_stage(curriculum_.stage_index());
  }
}

void APPOTrainer::apply_curriculum_lr() {
  float lr = curriculum_.learning_rate();
  for (auto& opt_group : actor_optimizer_.param_groups()) {
    opt_group.options().set_lr(lr);
  }
}

void APPOTrainer::maybe_initialize_from_checkpoint() {
  if (config_.ppo.init_checkpoint.empty()) {
    return;
  }
  const std::filesystem::path base(config_.ppo.init_checkpoint);
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_inference_checkpoint_metadata(metadata, checkpoint_config);

  const std::filesystem::path state_path = base / "state.pt";
  if (std::filesystem::exists(state_path)) {
    load_training_state(state_path);
    resumed_global_step_ = metadata.global_step;
    resumed_update_index_ = metadata.update_index;
  } else {
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
    }
    resumed_global_step_ = metadata.global_step;
    resumed_update_index_ = metadata.update_index;
  }

  if (metadata.extra.contains("wandb_run_id")) {
    config_.wandb.run_id = metadata.extra["wandb_run_id"].get<std::string>();
  }
  if (metadata.extra.contains("recent_scored_rates")) {
    recent_scored_rates_.clear();
    for (const auto& v : metadata.extra["recent_scored_rates"]) {
      recent_scored_rates_.push_back(v.get<double>());
    }
  }
  if (self_play_manager_ && self_play_manager_->enabled()) {
    if (metadata.extra.contains("self_play_rng_state")) {
      self_play_manager_->restore_rng_state(metadata.extra["self_play_rng_state"].get<std::string>());
    }
  }
  if (metadata.extra.contains("curriculum_stage") && curriculum_.enabled()) {
    CurriculumState restored;
    restored.stage_index = metadata.extra["curriculum_stage"].get<int>();
    restored.agent_steps_in_stage = metadata.extra.value("curriculum_agent_steps", 0LL);
    restored.promotion_counter = metadata.extra.value("curriculum_promotion_counter", 0);
    restored.demotion_counter = metadata.extra.value("curriculum_demotion_counter", 0);
    restored.current_mode = metadata.extra.value("curriculum_current_mode", std::string{"1v1"});
    if (metadata.extra.contains("curriculum_touch_rates")) {
      for (const auto& v : metadata.extra["curriculum_touch_rates"]) {
        restored.touch_rates.push_back(v.get<double>());
      }
    }
    if (metadata.extra.contains("curriculum_scored_rates")) {
      for (const auto& v : metadata.extra["curriculum_scored_rates"]) {
        restored.scored_rates.push_back(v.get<double>());
      }
    }
    curriculum_.restore_state(restored);
    std::cout << "restored_curriculum stage=" << restored.stage_index
              << " mode=" << restored.current_mode
              << " steps_in_stage=" << restored.agent_steps_in_stage << '\n';
  }
  if (log_initialization_) {
    std::cout << "initialized_from_checkpoint=" << base.string() << '\n';
  }
}

TrainerMetrics APPOTrainer::update_actor(RolloutStorage& rollout) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "update_actor");
#ifdef PULSAR_HAS_CUDA
  std::optional<c10::cuda::CUDAStream> _prev_train_stream;
  if (training_stream_.has_value()) {
    _prev_train_stream = c10::cuda::getCurrentCUDAStream(device_.index());
    c10::cuda::setCurrentCUDAStream(*training_stream_);
  }
#endif
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};

  float effective_entropy_coef = config_.ppo.entropy_coef;
  float effective_entropy_floor_coef = config_.ppo.entropy_floor_coef;
  if (config_.ppo.adaptive_entropy && !recent_scored_rates_.empty()) {
    double sum = 0.0;
    for (double v : recent_scored_rates_) sum += v;
    double recent_score = sum / static_cast<double>(recent_scored_rates_.size());
    double progress = std::clamp(
        (recent_score - config_.ppo.entropy_decay_score) /
            std::max(1.0e-6, 1.0 - config_.ppo.entropy_decay_score),
        0.0, 1.0);
    effective_entropy_coef = static_cast<float>(
        config_.ppo.entropy_coef + progress * (config_.ppo.entropy_low_coef - config_.ppo.entropy_coef));
    effective_entropy_floor_coef = config_.ppo.entropy_floor_coef * static_cast<float>(1.0 - progress);
  }

  const int seq_len = std::max(1, config_.ppo.rollout_length);
  const int logical_agents_per_batch = std::max(1, config_.ppo.minibatch_size / seq_len);
  const int max_forward_samples = std::max(1, config_.model.transformer_max_batch_size);
  const int agents_per_forward = std::max(1, max_forward_samples / seq_len);
  const int total_agents = rollout.num_agents();
  const int rollout_steps = rollout.rollout_length();
  std::int64_t metric_steps = 0;
  double accumulated_goal_critic_loss = 0.0;
  double accumulated_goal_score = 0.0;
  double accumulated_sampled_goal_distance = 0.0;

  const auto& all_values = rollout.all_values();
  const auto& all_rewards = rollout.all_rewards();
  if (rollout_steps <= 0) {
    metrics.effective_entropy_coef = static_cast<double>(effective_entropy_coef);
    return metrics;
  }
  const torch::Tensor extrinsic_values = all_values.at("extrinsic").narrow(0, 0, rollout_steps);
  const torch::Tensor extrinsic_rewards = all_rewards.at("extrinsic").narrow(0, 0, rollout_steps);
  const torch::Tensor rollout_dones = rollout.dones.narrow(0, 0, rollout_steps);
  const torch::Tensor rollout_bootstrap_truncated = rollout.bootstrap_truncated.narrow(0, 0, rollout_steps);

  torch::Tensor active_mask = rollout.learner_active.narrow(0, 0, rollout_steps) > 0.5F;
  torch::Tensor sparse_advantages;
  torch::Tensor normalized_advantages;
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "update_gae");
    torch::Tensor terminal_values;
    if (rollout_bootstrap_truncated.any().item<bool>()) {
      torch::NoGradGuard no_grad;
      torch::Tensor term_obs = rollout.terminal_observations.narrow(0, 0, rollout_steps);
      auto term_flat = term_obs.reshape({rollout_steps * total_agents, config_.model.observation_dim});
      const int total_term_samples = rollout_steps * total_agents;
      const int max_term_batch = std::max(1, config_.model.transformer_max_batch_size);
      std::vector<torch::Tensor> term_value_chunks;
      for (int offset = 0; offset < total_term_samples; offset += max_term_batch) {
        int batch = std::min(max_term_batch, total_term_samples - offset);
        auto chunk = term_flat.slice(0, offset, offset + batch).to(device_);
        auto chunk_goal = policy_goal_values_like(chunk, config_.goal_critic.goal_dim);
        auto chunk_out = actor_->forward_step(chunk, chunk_goal).value_win_logits.squeeze(-1);
        term_value_chunks.push_back(chunk_out.to(torch::kCPU));
      }
      auto term_values_flat = torch::cat(term_value_chunks, 0);
      terminal_values = term_values_flat.reshape({rollout_steps, total_agents}).to(term_obs.device());
    }
    sparse_advantages = compute_gae(
      extrinsic_values,
      extrinsic_rewards,
      rollout_dones,
      config_.ppo.gamma,
      config_.ppo.gae_lambda,
      rollout.final_values().count("extrinsic") ? rollout.final_values().at("extrinsic") : torch::Tensor{},
      rollout_bootstrap_truncated,
      terminal_values);
    normalized_advantages = normalize_advantage(sparse_advantages, active_mask);
  }
  torch::Tensor sparse_returns = sparse_advantages + extrinsic_values.detach();

  for (int epoch = 0; epoch < config_.ppo.update_epochs; ++epoch) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "update_epoch");
    const torch::Tensor perm = torch::randperm(total_agents, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    for (int agent_offset = 0; agent_offset < total_agents; agent_offset += logical_agents_per_batch) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "update_minibatch");
      const int count = std::min(logical_agents_per_batch, total_agents - agent_offset);
      const torch::Tensor agent_indices = perm.narrow(0, agent_offset, count);

      double total_active_samples_agent = 0.0;

      for (int seq_start = 0; seq_start < rollout.rollout_length(); seq_start += seq_len) {
        const int chunk_start = seq_start;
        const int chunk_end = std::min(rollout.rollout_length(), chunk_start + seq_len);
        const int chunk_steps = chunk_end - chunk_start;
        
        const int loss_start = chunk_start;
        const int loss_steps = chunk_steps;
        if (loss_steps <= 0) {
          continue;
        }
        total_active_samples_agent += rollout.learner_active
            .narrow(0, loss_start, loss_steps)
            .index_select(1, agent_indices)
            .sum()
            .item<double>();
      }
      if (total_active_samples_agent <= 0.0) {
        continue;
      }

      actor_optimizer_.zero_grad();

      for (int micro_agent_offset = 0; micro_agent_offset < count; micro_agent_offset += agents_per_forward) {
        PULSAR_TRACE_SCOPE_CAT("trainer", "update_microbatch");
        const int micro_count = std::min(agents_per_forward, count - micro_agent_offset);
        const torch::Tensor micro_agent_indices = agent_indices.narrow(0, micro_agent_offset, micro_count);

      for (int seq_start = 0; seq_start < rollout.rollout_length(); seq_start += seq_len) {
        const int chunk_start = seq_start;
        const int chunk_end = std::min(rollout.rollout_length(), chunk_start + seq_len);
        const int chunk_steps = chunk_end - chunk_start;
        
        const int loss_start = chunk_start;
        const int loss_steps = chunk_steps;

        const torch::Tensor obs =
            rollout.obs.narrow(0, chunk_start, chunk_steps).index_select(1, micro_agent_indices).to(device_);

        const auto forward_start = std::chrono::steady_clock::now();
        ActorSequenceOutput output;
        {
          PULSAR_TRACE_SCOPE_CAT("trainer", "update_forward_sequence");
          const torch::Tensor goal_values = policy_goal_values_like(obs, config_.goal_critic.goal_dim);
          output = actor_->forward_sequence(obs, goal_values);
        }

        if (loss_steps <= 0) {
          continue;
        }

        torch::Tensor policy_logits = output.policy_logits;
        torch::Tensor features = output.features;

        const torch::Tensor action_masks =
            rollout.action_masks.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices).to(device_).to(torch::kBool);
        const torch::Tensor learner_active =
            rollout.learner_active.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices).to(device_);
        const torch::Tensor old_actions =
            rollout.actions.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices).to(device_);
        const torch::Tensor old_log_probs =
            rollout.action_log_probs.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices).to(device_);
        const torch::Tensor chunk_advantages =
            normalized_advantages.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices).to(device_);

        const auto samples = loss_steps * micro_count;
        const torch::Tensor flat_active = learner_active.reshape({samples}) > 0.5F;
        if (flat_active.sum().item<std::int64_t>() == 0) {
          continue;
        }

        torch::Tensor flat_logits = policy_logits.reshape({samples, config_.model.action_dim});
        torch::Tensor flat_features = features.reshape({samples, static_cast<int64_t>(actor_->feature_dim())});
        torch::Tensor flat_masks = action_masks.reshape({samples, config_.model.action_dim});
        torch::Tensor flat_actions = old_actions.reshape({samples});
        torch::Tensor flat_old_log_probs = old_log_probs.reshape({samples});
        torch::Tensor flat_advantages = chunk_advantages.reshape({samples});

        const torch::Tensor active_logits = flat_logits.index({flat_active});
        const torch::Tensor active_features = flat_features.index({flat_active});
        const torch::Tensor active_masks = flat_masks.index({flat_active});
        const torch::Tensor active_actions = flat_actions.index({flat_active});
        const torch::Tensor active_old_log_probs = flat_old_log_probs.index({flat_active});
        const torch::Tensor active_advantages = flat_advantages.index({flat_active});

        const torch::Tensor log_probs =
            torch::log_softmax(apply_action_mask_to_logits(active_logits, active_masks), -1);
        const torch::Tensor current_log_probs = log_probs.gather(1, active_actions.unsqueeze(1)).squeeze(1);

        torch::Tensor epsilon = torch::full({active_advantages.size(0)}, config_.ppo.clip_range, active_advantages.options());

        torch::Tensor policy_loss =
            clipped_ppo_policy_loss(current_log_probs, active_old_log_probs, active_advantages, epsilon);
        policy_loss = policy_loss.mean();

        const torch::Tensor entropy = masked_action_entropy(active_logits, active_masks).mean();
        torch::Tensor entropy_floor_loss = torch::zeros({}, active_advantages.options());
        if (config_.ppo.entropy_floor > 0.0F && effective_entropy_floor_coef > 0.0F) {
          const torch::Tensor entropy_floor = torch::full_like(entropy, config_.ppo.entropy_floor);
          entropy_floor_loss = effective_entropy_floor_coef * torch::relu(entropy_floor - entropy).square();
        }

        torch::Tensor chunk_returns =
            sparse_returns.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices).to(device_).reshape({samples});
        torch::Tensor active_returns = chunk_returns.index({flat_active});

        torch::Tensor value_win_chunk = output.value_win_logits;
        torch::Tensor flat_value_win_logits = value_win_chunk.reshape({samples, 1});
        torch::Tensor active_value_win_logits = flat_value_win_logits.index({flat_active});

        torch::Tensor value_loss = torch::mse_loss(
            active_value_win_logits.squeeze(-1), active_returns, torch::Reduction::Mean);

        torch::Tensor goal_loss = torch::zeros({}, active_advantages.options());
        torch::Tensor actor_goal_loss = torch::zeros({}, active_advantages.options());
        double chunk_goal_score = 0.0;

        {
          torch::Tensor chunk_goal_pos =
              rollout.goal_positions.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices);
          torch::Tensor chunk_dones =
              rollout.dones.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices);
          torch::Tensor chunk_ep_starts =
              rollout.episode_starts.narrow(0, loss_start, loss_steps).index_select(1, micro_agent_indices);
          torch::Tensor future_goal_pos = sample_future_goal_positions(
              chunk_goal_pos,
              chunk_dones,
              chunk_ep_starts,
              config_.goal_critic.max_future_horizon);
          const int goal_dim = config_.goal_critic.goal_dim;

          torch::Tensor flat_future_goal_pos = future_goal_pos.to(device_).reshape({samples, goal_dim});
          torch::Tensor active_future_goal_pos = flat_future_goal_pos.index({flat_active});

          const auto active_count = active_features.size(0);
          const int cb_size = config_.goal_critic.contrastive_batch_size;
          torch::Tensor sa_emb, g_emb;
          if (active_count > static_cast<c10::IntArrayRef::value_type>(cb_size)) {
            const torch::Tensor idx = torch::randperm(active_count, active_actions.options())
                .narrow(0, 0, cb_size);
            const torch::Tensor feat_sub = active_features.index({idx});
            const torch::Tensor act_sub = active_actions.index({idx});
            const torch::Tensor goal_sub = active_future_goal_pos.index({idx});
            const torch::Tensor logit_sub = active_logits.index({idx});
            const torch::Tensor mask_sub = active_masks.index({idx});

            sa_emb = actor_->goal_critic()->sa_embedding(feat_sub, act_sub);
            g_emb = actor_->goal_critic()->goal_embedding(goal_sub);

            torch::Tensor sampled = sample_masked_actions(
                logit_sub.detach(), mask_sub.detach(), false, nullptr);
            actor_goal_loss = -actor_->goal_critic()->forward(
                feat_sub, sampled.detach(), goal_sub).mean();
          } else {
            sa_emb = actor_->goal_critic()->sa_embedding(active_features, active_actions);
            g_emb = actor_->goal_critic()->goal_embedding(active_future_goal_pos);

            torch::Tensor sampled = sample_masked_actions(
                active_logits.detach(), active_masks.detach(), false, nullptr);
            actor_goal_loss = -actor_->goal_critic()->forward(
                active_features, sampled.detach(), active_future_goal_pos).mean();
          }
          const torch::Tensor sa_logits = compute_pairwise_negative_l2_logits(sa_emb, g_emb);
          goal_loss = compute_symmetric_infonce_loss(sa_logits, config_.goal_critic.logsumexp_penalty_coeff);

          {
            torch::NoGradGuard no_grad;
            const torch::Tensor goal_scores = actor_->goal_critic()->forward(
                active_features.detach(),
                active_actions,
                active_future_goal_pos);
            chunk_goal_score = goal_scores.mean().item<double>();
          }

          const double chunk_sampled_goal_norm = active_future_goal_pos.norm(2, -1).mean().item<double>();
          accumulated_sampled_goal_distance += chunk_sampled_goal_norm * static_cast<double>(active_logits.size(0));
          accumulated_goal_critic_loss += goal_loss.item<double>() * static_cast<double>(active_logits.size(0));
          accumulated_goal_score += chunk_goal_score * static_cast<double>(active_logits.size(0));
        }

        const auto active_samples = static_cast<double>(active_logits.size(0));
        const auto sample_weight = active_samples / total_active_samples_agent;

        if (config_.ppo.pcgrad) {
          torch::Tensor loss_a = (policy_loss
              + config_.ppo.value_coef * value_loss
              + entropy_floor_loss
              - effective_entropy_coef * entropy) * sample_weight;

          std::vector<torch::Tensor> saved_grads;
          for (auto& p : actor_->parameters()) {
            saved_grads.push_back(p.grad().defined() ? p.grad().clone() : torch::Tensor{});
          }

          zero_existing_gradients(*actor_);
          loss_a.backward({}, true);
          auto grads_a = capture_gradients(*actor_);

          torch::Tensor loss_b = (config_.goal_critic.lambda_Zg * goal_loss
              + config_.goal_critic.lambda_goal_actor * actor_goal_loss) * sample_weight;

          zero_existing_gradients(*actor_);
          loss_b.backward();
          auto grads_b = capture_gradients(*actor_);
          zero_existing_gradients(*actor_);

          apply_pcgrad(grads_a, grads_b);

          for (size_t i = 0; i < grads_a.size(); ++i) {
            torch::Tensor combined;
            bool has_a = grads_a[i].grad.defined();
            bool has_b = grads_b[i].grad.defined();
            if (has_a && has_b) {
              combined = grads_a[i].grad + grads_b[i].grad;
            } else if (has_a) {
              combined = grads_a[i].grad;
            } else if (has_b) {
              combined = grads_b[i].grad;
            } else {
              // No gradient from either group this micro-batch — restore
              // any previously accumulated gradient so it isn't dropped.
              if (saved_grads[i].defined()) {
                grads_a[i].param.mutable_grad() = saved_grads[i];
              }
              continue;
            }
            if (saved_grads[i].defined()) {
              grads_a[i].param.mutable_grad() = saved_grads[i] + combined;
            } else {
              grads_a[i].param.mutable_grad() = combined;
            }
          }
        } else {
          const torch::Tensor loss =
              policy_loss
              + config_.ppo.value_coef * value_loss
              + config_.goal_critic.lambda_Zg * goal_loss
              + config_.goal_critic.lambda_goal_actor * actor_goal_loss
              + entropy_floor_loss
              - effective_entropy_coef * entropy;

          torch::Tensor combined_loss = loss * sample_weight;

          combined_loss.backward();
        }

        metrics.policy_loss += policy_loss.item<double>() * static_cast<double>(active_samples);
        metrics.value_loss += value_loss.item<double>() * static_cast<double>(active_samples);
        metrics.entropy += entropy.item<double>() * static_cast<double>(active_samples);
        metric_steps += active_samples;
      }
      }

      const auto optim_start = std::chrono::steady_clock::now();
      double grad_norm = 0.0;
      {
        PULSAR_TRACE_SCOPE_CAT("trainer", "update_optimizer");
        const auto grad_norm_value = torch::nn::utils::clip_grad_norm_(actor_->parameters(), config_.ppo.max_grad_norm);
        grad_norm = static_cast<double>(grad_norm_value);
        actor_optimizer_.step();
      }
      metrics.optimizer_step_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - optim_start).count();
      metrics.grad_norm += grad_norm * total_active_samples_agent;
    }
  }

  if (metric_steps > 0) {
    metrics.policy_loss /= static_cast<double>(metric_steps);
    metrics.value_loss /= static_cast<double>(metric_steps);
    metrics.entropy /= static_cast<double>(metric_steps);
    metrics.grad_norm /= static_cast<double>(metric_steps);
    metrics.goal_critic_loss = accumulated_goal_critic_loss / static_cast<double>(metric_steps);
    metrics.mean_goal_score = accumulated_goal_score / static_cast<double>(metric_steps);
    metrics.mean_sampled_goal_distance = accumulated_sampled_goal_distance / static_cast<double>(metric_steps);
  }
  metrics.effective_entropy_coef = static_cast<double>(effective_entropy_coef);
  metrics.update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  return metrics;
}

APPOTrainer::ESPopulationFitness APPOTrainer::evaluate_es_population(
    const torch::Tensor& A_stack,
    const torch::Tensor& B_stack,
    int update_index) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "es_evaluate");
  torch::NoGradGuard no_grad_guard;
  const auto& es_cfg = config_.es_lora;
  const int pop = es_cfg.population_size;
  const int eval_envs = es_cfg.eval_num_envs;
  const int total_envs = pop * eval_envs;
  const int team_size = config_.env.team_size;
  const int agents_per_env = team_size * 2;
  const int member_agents = eval_envs * agents_per_env;

  ESPopulationFitness result;
  result.fitness.assign(static_cast<std::size_t>(pop), 0.0F);
  result.winrate.assign(static_cast<std::size_t>(pop), 0.0F);
  result.kl.assign(static_cast<std::size_t>(pop), 0.0F);

  std::vector<int> episode_counts(static_cast<std::size_t>(pop), 0);
  std::vector<int> win_counts(static_cast<std::size_t>(pop), 0);

  torch::Tensor kl_sum = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor kl_count = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  std::vector<std::uint8_t> controlled_host(static_cast<std::size_t>(total_envs * agents_per_env), 0);
  for (int env_idx = 0; env_idx < total_envs; ++env_idx) {
    const int local_env = env_idx % eval_envs;
    const bool perturb_blue = (local_env % 2) == 0;
    for (int local_agent = 0; local_agent < agents_per_env; ++local_agent) {
      const bool is_blue = local_agent < team_size;
      controlled_host[static_cast<std::size_t>(env_idx * agents_per_env + local_agent)] =
          (is_blue == perturb_blue) ? 1 : 0;
    }
  }
  const torch::Tensor controlled_mask = torch::from_blob(
      controlled_host.data(),
      {static_cast<long>(controlled_host.size())},
      torch::TensorOptions().dtype(torch::kUInt8))
      .clone()
      .to(device_)
      .to(torch::kBool);
  const torch::Tensor controlled_float = controlled_mask.to(torch::kFloat32).view({pop, member_agents});

  auto eval_collector = make_es_eval_collector(
      config_, total_envs, eval_envs, update_index, 0, use_pinned_host_buffers_);

  for (int ep = 0; ep < es_cfg.eval_episodes_per_member; ++ep) {
    eval_collector->reset_es_episode(update_index, ep, eval_envs);

    for (int step = 0; step < es_cfg.eval_rollout_length; ++step) {
      torch::Tensor raw_obs = eval_collector->host_observations().to(device_, use_pinned_host_buffers_);
      torch::Tensor episode_starts = eval_collector->host_episode_starts().to(device_, use_pinned_host_buffers_);
      torch::Tensor action_masks = eval_collector->host_action_masks().to(device_, use_pinned_host_buffers_).to(torch::kBool);
      torch::Tensor normalized_obs = actor_normalizer_.normalize(raw_obs);

      const torch::Tensor goal_values = policy_goal_values_like(normalized_obs, config_.goal_critic.goal_dim);
      ActorStepOutput output = actor_->forward_step(normalized_obs, goal_values);
      torch::Tensor perturbed_logits = actor_->policy_eggroll_logits(
          output.features, A_stack, B_stack, es_cfg.sigma_ES, goal_values);

      torch::Tensor base_actions = sample_masked_actions(output.policy_logits, action_masks, true, nullptr);
      torch::Tensor perturbed_actions = sample_masked_actions(perturbed_logits, action_masks, true, nullptr);
      torch::Tensor actions = torch::where(controlled_mask, perturbed_actions, base_actions);

      const torch::Tensor base_masked = apply_action_mask_to_logits(output.policy_logits, action_masks);
      const torch::Tensor perturbed_masked = apply_action_mask_to_logits(perturbed_logits, action_masks);
      const torch::Tensor base_probs = torch::softmax(base_masked, -1);
      const torch::Tensor perturbed_probs = torch::softmax(perturbed_masked, -1);
      const torch::Tensor kl_values = (
          perturbed_probs * (torch::log(perturbed_probs + 1.0e-8) - torch::log(base_probs + 1.0e-8)))
          .sum(-1)
          .view({pop, member_agents});
      kl_sum += (kl_values * controlled_float).sum(1);
      kl_count += controlled_float.sum(1);

      const torch::Tensor action_indices_cpu = actions.contiguous().to(torch::kCPU);
      eval_collector->step(std::span<const std::int64_t>(
          action_indices_cpu.data_ptr<std::int64_t>(),
          static_cast<std::size_t>(action_indices_cpu.numel())));

      torch::Tensor dones_cpu = eval_collector->host_dones().to(torch::kCPU);
      torch::Tensor labels_cpu = eval_collector->host_terminal_outcome_labels().to(torch::kCPU);
      const auto* dones_ptr = dones_cpu.data_ptr<float>();
      const auto* labels_ptr = labels_cpu.data_ptr<std::int64_t>();
      for (std::size_t i = 0; i < controlled_host.size(); ++i) {
        if (controlled_host[i] == 0 || dones_ptr[i] <= 0.5F) {
          continue;
        }
        const int env_idx = static_cast<int>(i / static_cast<std::size_t>(agents_per_env));
        const int member = env_idx / eval_envs;
        episode_counts[static_cast<std::size_t>(member)] += 1;
        if (labels_ptr[i] == 0) {
          win_counts[static_cast<std::size_t>(member)] += 1;
        }
    }
  }
}

  torch::Tensor kl_mean = (kl_sum / kl_count.clamp_min(1.0F)).to(torch::kCPU);
  const auto* kl_ptr = kl_mean.data_ptr<float>();
  for (int i = 0; i < pop; ++i) {
    const int denom = std::max(episode_counts[static_cast<std::size_t>(i)], 1);
    result.winrate[static_cast<std::size_t>(i)] =
        static_cast<float>(win_counts[static_cast<std::size_t>(i)]) / static_cast<float>(denom);
    result.kl[static_cast<std::size_t>(i)] = kl_ptr[i];
    result.fitness[static_cast<std::size_t>(i)] =
        result.winrate[static_cast<std::size_t>(i)]
        - es_cfg.beta_KL * result.kl[static_cast<std::size_t>(i)];
  }
  return result;
}

void APPOTrainer::run_es_lora_update(int update_index, TrainerMetrics& metrics) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "es_update");
  const auto es_start = std::chrono::steady_clock::now();
  const auto& es_cfg = config_.es_lora;
  const int pop = es_cfg.population_size;
  const int rank = es_cfg.rank;
  const int in_features = actor_->policy_lora()->in_features();
  const int out_features = actor_->policy_lora()->out_features();

  const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor A_stack;
  torch::Tensor B_stack;
  if (es_cfg.antithetic_sampling) {
    const int half_pop = pop / 2;
    torch::Tensor A_half = torch::randn({half_pop, rank, in_features}, tensor_options);
    torch::Tensor B_half = torch::randn({half_pop, out_features, rank}, tensor_options);
    A_stack = torch::cat({A_half, -A_half}, 0);
    B_stack = torch::cat({B_half, -B_half}, 0);
  } else {
    A_stack = torch::randn({pop, rank, in_features}, tensor_options);
    B_stack = torch::randn({pop, out_features, rank}, tensor_options);
  }

  ESPopulationFitness population = evaluate_es_population(A_stack, B_stack, update_index);
  std::vector<float>& fitnesses = population.fitness;
  const uint64_t total_members = fitnesses.size();
  float mu = 0.0F;
  for (float f : fitnesses) {
    mu += f;
  }
  mu /= static_cast<float>(total_members);

  float sigma = 0.0F;
  for (float f : fitnesses) {
    sigma += (f - mu) * (f - mu);
  }
  sigma = std::sqrt(sigma / static_cast<float>(total_members));

  std::vector<float> normalized_f;
  for (float f : fitnesses) {
    normalized_f.push_back((f - mu) / (sigma + 1.0e-8F));
  }

  double winrate_mean = 0.0;
  double kl_mean = 0.0;
  for (uint64_t i = 0; i < total_members; ++i) {
    winrate_mean += population.winrate[i];
    kl_mean += population.kl[i];
  }
  winrate_mean /= static_cast<double>(total_members);
  kl_mean /= static_cast<double>(total_members);

  double winrate_variance = 0.0;
  for (uint64_t i = 0; i < total_members; ++i) {
    const double centered = static_cast<double>(population.winrate[i]) - winrate_mean;
    winrate_variance += centered * centered;
  }
  const double winrate_std = std::sqrt(winrate_variance / static_cast<double>(total_members));
  const float best_fitness = *std::max_element(fitnesses.begin(), fitnesses.end());
  if (es_cfg.require_winrate_signal && winrate_std < static_cast<double>(es_cfg.min_winrate_std)) {
    metrics.es_fitness_mean = mu;
    metrics.es_fitness_std = sigma;
    metrics.es_fitness_best = static_cast<double>(best_fitness);
    metrics.es_update_norm = 0.0;
    metrics.es_winrate_mean = winrate_mean;
    metrics.es_kl_mean = kl_mean;

    auto lora_params = actor_->es_lora_parameters();
    metrics.es_lora_a_norm = static_cast<double>(lora_params[0].norm().item<float>());
    metrics.es_lora_b_norm = static_cast<double>(lora_params[1].norm().item<float>());
    metrics.es_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - es_start).count();
    return;
  }

  torch::Tensor grad_A = torch::zeros(
      {rank, in_features},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor grad_B = torch::zeros(
      {out_features, rank},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  for (uint64_t i = 0; i < total_members; ++i) {
    grad_A.add_(A_stack[static_cast<long>(i)], normalized_f[i]);
    grad_B.add_(B_stack[static_cast<long>(i)], normalized_f[i]);
  }
  grad_A.div_(static_cast<float>(total_members));
  grad_B.div_(static_cast<float>(total_members));

  const float step = es_cfg.eta_ES / es_cfg.sigma_ES;
  torch::Tensor delta_A = grad_A * step;
  torch::Tensor delta_B = grad_B * step;

  double update_norm = std::sqrt(
      std::pow(static_cast<double>(delta_A.norm().item<float>()), 2) +
      std::pow(static_cast<double>(delta_B.norm().item<float>()), 2));
  double update_scale = 1.0;
  if (es_cfg.max_kl_mean > 0.0F && kl_mean > static_cast<double>(es_cfg.max_kl_mean)) {
    update_scale = std::min(update_scale, static_cast<double>(es_cfg.max_kl_mean) / std::max(kl_mean, 1.0e-12));
  }
  if (es_cfg.update_norm_clip && es_cfg.max_update_norm > 0.0F && update_norm > static_cast<double>(es_cfg.max_update_norm)) {
    update_scale = std::min(update_scale, static_cast<double>(es_cfg.max_update_norm) / std::max(update_norm, 1.0e-12));
  }
  if (update_scale < 1.0) {
    delta_A.mul_(update_scale);
    delta_B.mul_(update_scale);
    update_norm *= update_scale;
  }

  {
    torch::NoGradGuard no_grad;
    auto lora_params = actor_->es_lora_parameters();
    lora_params[0].add_(delta_A);
    lora_params[1].add_(delta_B);
  }

  metrics.es_fitness_mean = mu;
  metrics.es_fitness_std = sigma;
  metrics.es_fitness_best = static_cast<double>(best_fitness);
  metrics.es_update_norm = update_norm;
  metrics.es_winrate_mean = winrate_mean;
  metrics.es_kl_mean = kl_mean;

  auto lora_params = actor_->es_lora_parameters();
  metrics.es_lora_a_norm = static_cast<double>(lora_params[0].norm().item<float>());
  metrics.es_lora_b_norm = static_cast<double>(lora_params[1].norm().item<float>());

  metrics.es_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - es_start).count();
}

void APPOTrainer::collect_rollout(
    RolloutStorage& dest,
    TrainerMetrics& metrics,
    std::int64_t* collected_agent_steps,
    PPOActor rollout_actor) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "collect_rollout");
  if (!rollout_actor) {
    throw std::invalid_argument("APPOTrainer::collect_rollout requires a policy snapshot.");
  }
  dest.clear();
  const auto update_start = std::chrono::steady_clock::now();
  CollectorTimings collector_timings{};
  BatchedRocketSimCollector* collector_ = collectors_.front().get();
  std::int64_t local_collected_steps = 0;

  const auto collection_start = std::chrono::steady_clock::now();

  double total_reward = 0.0;
  double total_gameplay_reward = 0.0;
  double total_mechanic_reward = 0.0;
  int64_t total_steps = 0;
  int64_t total_learner_steps = 0;
  double accumulated_sampled_value = 0.0;
  int64_t accumulated_value_count = 0;
  double total_goal_distance = 0.0;
  double min_goal_distance = 1.0;
  int64_t total_goals_scored = 0;
  int64_t total_goals_conceded = 0;
  int64_t total_ball_proximity_steps = 0;
  int64_t total_ball_proximity_denom = 0;
  int completed_episodes = 0;
  int scored_episodes = 0;
  int touched_episodes = 0;
  const int agents_per_env = std::max(1, config_.env.team_size * 2);

  if (collectors_.size() > 1) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "collect_loop_sharded");
    struct PendingShardStep {
      int agent_offset = 0;
      std::size_t shard = 0;
      torch::Tensor normalized_obs{};
      torch::Tensor episode_starts_host{};
      torch::Tensor action_masks_host{};
      torch::Tensor learner_active_host{};
      torch::Tensor action_indices_cpu{};
      torch::Tensor action_log_probs{};
      torch::Tensor sampled_value{};
      CollectorTimings timings{};
      std::future<void> step_future{};
    };

    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "collect_step_sharded");
      std::vector<PendingShardStep> pending;
      pending.reserve(collectors_.size());

      for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
        auto& collector = *collectors_[shard];
        torch::Tensor raw_obs_host = collector.host_observations();
        torch::Tensor episode_starts_host = collector.host_episode_starts();
        torch::Tensor action_masks_host = collector.host_action_masks();
        torch::Tensor learner_active_host = collector.host_learner_active();
        torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
        torch::Tensor episode_starts = episode_starts_host.to(device_, use_pinned_host_buffers_);
        torch::Tensor action_masks = action_masks_host.to(device_, use_pinned_host_buffers_).to(torch::kBool);

        torch::Tensor normalized_obs;
        torch::Tensor actions;
        torch::Tensor action_log_probs;
        ActorStepOutput output;
        const auto policy_start = std::chrono::steady_clock::now();
        {
          PULSAR_TRACE_SCOPE_CAT("trainer", "policy_forward_shard");
          torch::NoGradGuard no_grad;
          actor_normalizer_.update(raw_obs);
          normalized_obs = actor_normalizer_.normalize(raw_obs);
          const torch::Tensor goal_values = policy_goal_values_like(normalized_obs, config_.goal_critic.goal_dim);
          output = rollout_actor->forward_step(normalized_obs, goal_values);
          actions = sample_masked_actions(output.policy_logits, action_masks, false, &action_log_probs);
        }
        if (config_.ppo.synchronize_cuda_timing && device_.is_cuda()) {
          torch::cuda::synchronize();
        }
        metrics.policy_forward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

        if (self_play_manager_ && self_play_manager_->has_snapshots()) {
          torch::Tensor opponent_actions;
          torch::Tensor snapshot_ids = collector.host_snapshot_ids().to(device_, use_pinned_host_buffers_);
          self_play_manager_->infer_opponent_actions(
              rollout_actor,
              raw_obs,
              action_masks,
              episode_starts,
              snapshot_ids,
              &opponent_actions,
              &metrics.policy_forward_seconds);
          actions = torch::where(snapshot_ids >= 0, opponent_actions, actions);
        }

        pending.emplace_back();
        PendingShardStep& shard_step = pending.back();
        shard_step.agent_offset = static_cast<int>(shard_agent_offsets_[shard]);
        shard_step.shard = shard;
        shard_step.normalized_obs = normalized_obs;
        shard_step.episode_starts_host = episode_starts_host;
        shard_step.action_masks_host = action_masks_host;
        shard_step.learner_active_host = learner_active_host;
        shard_step.action_log_probs = action_log_probs;
        shard_step.sampled_value = output.value_win_logits.squeeze(-1);

        const auto decode_start = std::chrono::steady_clock::now();
        {
          PULSAR_TRACE_SCOPE_CAT("trainer", "action_decode_shard");
          shard_step.action_indices_cpu = actions.contiguous().to(torch::kCPU);
          BatchedRocketSimCollector* collector_ptr = &collector;
          torch::Tensor action_indices_cpu = shard_step.action_indices_cpu;
          CollectorTimings* shard_timings = &shard_step.timings;
          shard_step.step_future = std::async(
              std::launch::async,
              [collector_ptr, action_indices_cpu, shard_timings]() mutable {
                PULSAR_TRACE_SCOPE_CAT("trainer", "async_step_shard");
                collector_ptr->step(
                    std::span<const std::int64_t>(
                        action_indices_cpu.data_ptr<std::int64_t>(),
                        static_cast<std::size_t>(action_indices_cpu.numel())),
                    shard_timings);
              });
        }
        metrics.action_decode_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
      }

      for (PendingShardStep& shard_step : pending) {
        shard_step.step_future.get();
        accumulate_timings(collector_timings, shard_step.timings);

        auto& collector = *collectors_[shard_step.shard];
        torch::Tensor dones_host = collector.host_dones();
        torch::Tensor truncated_host = collector.host_truncated();
        torch::Tensor bootstrap_truncated_host = collector.host_bootstrap_truncated();
        torch::Tensor terminal_labels = collector.host_terminal_outcome_labels();
        torch::Tensor extrinsic_rewards_host = collector.host_rewards();
        torch::Tensor gameplay_r_host = collector.host_gameplay_rewards();
        torch::Tensor mechanic_r_host = collector.host_mechanic_rewards();
        const auto* dones_ptr = dones_host.data_ptr<float>();

        torch::Tensor ball_prox_host = collector.host_ball_proximity();
        total_ball_proximity_steps += ball_prox_host.sum().item<int64_t>();
        total_ball_proximity_denom += ball_prox_host.numel();

        const auto* tl_ptr = terminal_labels.data_ptr<std::int64_t>();
        const auto* la_ptr = shard_step.learner_active_host.data_ptr<float>();
        torch::Tensor env_touch_host = collector.host_env_touched();
        const auto* env_touch_ptr = env_touch_host.data_ptr<float>();
        for (int64_t env_agent_begin = 0; env_agent_begin < terminal_labels.numel(); env_agent_begin += agents_per_env) {
          const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + agents_per_env, terminal_labels.numel());
          bool env_done = false;
          bool env_scored = false;
          bool env_conceded = false;
          for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
            if (la_ptr[i] > 0.5F && dones_ptr[i] > 0.5F) {
              env_done = true;
              if (tl_ptr[i] == 0) env_scored = true;
              if (tl_ptr[i] == 1) env_conceded = true;
            }
          }
          if (env_done) {
            if (env_scored) total_goals_scored++;
            if (env_conceded) total_goals_conceded++;
          }
        }
        for (int64_t env_agent_begin = 0; env_agent_begin < dones_host.numel(); env_agent_begin += agents_per_env) {
          bool env_done = false;
          bool env_scored = false;
          const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + agents_per_env, dones_host.numel());
          for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
            env_done = env_done || dones_ptr[i] > 0.5F;
            env_scored = env_scored || (dones_ptr[i] > 0.5F && (tl_ptr[i] == 0 || tl_ptr[i] == 1));
          }
          if (env_done) {
             completed_episodes++;
             if (env_scored) {
               scored_episodes++;
             }
             const int64_t env_idx = env_agent_begin / agents_per_env;
             if (env_touch_ptr[env_idx] > 0.5F) {
               touched_episodes++;
             }
           }
         }

         const torch::Tensor sampled_value_cpu = shard_step.sampled_value.to(torch::kCPU);
        accumulated_sampled_value += sampled_value_cpu.sum().item<double>();
        accumulated_value_count += static_cast<int64_t>(sampled_value_cpu.numel());

        torch::Tensor goal_pos_host = collector.host_goal_positions();
        torch::Tensor goal_norms = goal_pos_host.norm(2, 1);
        float gd_min = goal_norms.min().item<float>();
        float gd_mean = goal_norms.mean().item<float>();
        total_goal_distance += static_cast<double>(gd_mean)
            * static_cast<double>(goal_pos_host.size(0))
            / static_cast<double>(total_agents_);
        if (gd_min < min_goal_distance) {
          min_goal_distance = static_cast<double>(gd_min);
        }

        torch::Tensor terminal_obs_host = collector.host_terminal_observations();

        const auto learner_step_count = static_cast<std::int64_t>(shard_step.learner_active_host.sum().item<float>());
        total_reward += extrinsic_rewards_host.sum().item<double>();
        total_gameplay_reward += gameplay_r_host.sum().item<double>();
        total_mechanic_reward += mechanic_r_host.sum().item<double>();
        total_steps += extrinsic_rewards_host.numel();
        total_learner_steps += learner_step_count;

        std::unordered_map<std::string, torch::Tensor> all_values;
        all_values["extrinsic"] = sampled_value_cpu;

        std::unordered_map<std::string, torch::Tensor> all_rewards;
        all_rewards["extrinsic"] = extrinsic_rewards_host;

        dest.append_slice(
            step,
            shard_step.agent_offset,
            shard_step.normalized_obs.to(torch::kCPU),
            shard_step.episode_starts_host.to(torch::kBool),
            shard_step.action_masks_host,
            shard_step.learner_active_host,
            shard_step.action_indices_cpu,
            shard_step.action_log_probs.to(torch::kCPU),
            all_values,
            all_rewards,
            dones_host,
            truncated_host,
            bootstrap_truncated_host,
            goal_pos_host,
            terminal_labels,
            terminal_obs_host);

        local_collected_steps += learner_step_count;
      }

    }

    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "bootstrap_forward_sharded");
      torch::NoGradGuard no_grad;
      std::vector<torch::Tensor> final_values;
      final_values.reserve(collectors_.size());
      for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
        auto& collector = *collectors_[shard];
        torch::Tensor final_raw_obs = collector.host_observations().to(device_, use_pinned_host_buffers_);
        torch::Tensor final_normalized = actor_normalizer_.normalize(final_raw_obs);
        torch::Tensor final_starts = collector.host_episode_starts().to(device_, use_pinned_host_buffers_);
        const torch::Tensor final_goal_values = policy_goal_values_like(final_normalized, config_.goal_critic.goal_dim);
        ActorStepOutput final_output = rollout_actor->forward_step(final_normalized, final_goal_values);
        final_values.push_back(final_output.value_win_logits.squeeze(-1).to(torch::kCPU));
      }
      std::unordered_map<std::string, torch::Tensor> bootstrap_values;
      bootstrap_values["extrinsic"] = torch::cat(final_values, 0);
      dest.set_final_values(bootstrap_values);
    }
  } else {
    PULSAR_TRACE_SCOPE_CAT("trainer", "collect_loop");
    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "collect_step");
      torch::Tensor raw_obs_host = collector_->host_observations();
    torch::Tensor episode_starts_host = collector_->host_episode_starts();
    torch::Tensor action_masks_host = collector_->host_action_masks();
    torch::Tensor learner_active_host = collector_->host_learner_active();
    torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
    torch::Tensor episode_starts = episode_starts_host.to(device_, use_pinned_host_buffers_);
    torch::Tensor action_masks = action_masks_host.to(device_, use_pinned_host_buffers_).to(torch::kBool);

    torch::Tensor normalized_obs;
    torch::Tensor actions;
    torch::Tensor action_log_probs;
    ActorStepOutput output;
    const auto policy_start = std::chrono::steady_clock::now();
    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "policy_forward");
      torch::NoGradGuard no_grad;
      actor_normalizer_.update(raw_obs);
      normalized_obs = actor_normalizer_.normalize(raw_obs);
      const torch::Tensor goal_values = policy_goal_values_like(normalized_obs, config_.goal_critic.goal_dim);
      output = rollout_actor->forward_step(normalized_obs, goal_values);
      actions = sample_masked_actions(output.policy_logits, action_masks, false, &action_log_probs);
    }
    if (config_.ppo.synchronize_cuda_timing && device_.is_cuda()) {
      torch::cuda::synchronize();
    }
    metrics.policy_forward_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

    if (self_play_manager_ && self_play_manager_->has_snapshots()) {
      torch::Tensor opponent_actions;
      torch::Tensor snapshot_ids = collector_->host_snapshot_ids().to(device_, use_pinned_host_buffers_);
      self_play_manager_->infer_opponent_actions(
          rollout_actor,
          raw_obs,
          action_masks,
          episode_starts,
          snapshot_ids,
          &opponent_actions,
          &metrics.policy_forward_seconds);
      actions = torch::where(snapshot_ids >= 0, opponent_actions, actions);
    }

    torch::Tensor sampled_value = output.value_win_logits.squeeze(-1);

    const auto decode_start = std::chrono::steady_clock::now();
    torch::Tensor action_indices_cpu;
    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "action_decode");
      action_indices_cpu = actions.contiguous().to(torch::kCPU);
      collector_->step(
          std::span<const std::int64_t>(
              action_indices_cpu.data_ptr<std::int64_t>(),
              static_cast<std::size_t>(action_indices_cpu.numel())),
          &collector_timings);
    }
    metrics.action_decode_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();

    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "collect_post_step");
      torch::Tensor dones_host = collector_->host_dones();
    torch::Tensor truncated_host = collector_->host_truncated();
    torch::Tensor bootstrap_truncated_host = collector_->host_bootstrap_truncated();
    torch::Tensor terminal_labels = collector_->host_terminal_outcome_labels();
      torch::Tensor extrinsic_rewards_host = collector_->host_rewards();
      torch::Tensor gameplay_r_host = collector_->host_gameplay_rewards();
      torch::Tensor mechanic_r_host = collector_->host_mechanic_rewards();
    const auto* dones_ptr = dones_host.data_ptr<float>();

    torch::Tensor ball_prox_host = collector_->host_ball_proximity();
    total_ball_proximity_steps += ball_prox_host.sum().item<int64_t>();
    total_ball_proximity_denom += ball_prox_host.numel();

    const auto* tl_ptr = terminal_labels.data_ptr<std::int64_t>();
    const auto* la_ptr = learner_active_host.data_ptr<float>();
    torch::Tensor env_touch_host = collector_->host_env_touched();
    const auto* env_touch_ptr = env_touch_host.data_ptr<float>();
    for (int64_t env_agent_begin = 0; env_agent_begin < terminal_labels.numel(); env_agent_begin += agents_per_env) {
      const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + agents_per_env, terminal_labels.numel());
      bool env_goal_scored = false;
      bool env_goal_conceded = false;
      for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
        if (la_ptr[i] > 0.5F && dones_ptr[i] > 0.5F) {
          if (tl_ptr[i] == 0) env_goal_scored = true;
          if (tl_ptr[i] == 1) env_goal_conceded = true;
        }
      }
      if (env_goal_scored) total_goals_scored++;
      if (env_goal_conceded) total_goals_conceded++;
    }
    for (int64_t env_agent_begin = 0; env_agent_begin < dones_host.numel(); env_agent_begin += agents_per_env) {
      bool env_done = false;
      bool env_scored = false;
      const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + agents_per_env, dones_host.numel());
      for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
        env_done = env_done || dones_ptr[i] > 0.5F;
        env_scored = env_scored || (dones_ptr[i] > 0.5F && (tl_ptr[i] == 0 || tl_ptr[i] == 1));
      }
      if (env_done) {
        completed_episodes++;
        if (env_scored) {
          scored_episodes++;
        }
        const int64_t env_idx = env_agent_begin / agents_per_env;
        if (env_touch_ptr[env_idx] > 0.5F) {
          touched_episodes++;
        }
      }
    }

    const torch::Tensor sampled_value_cpu = sampled_value.to(torch::kCPU);
    accumulated_sampled_value += sampled_value_cpu.sum().item<double>();
    accumulated_value_count += static_cast<int64_t>(sampled_value_cpu.numel());

    torch::Tensor goal_pos_host = collector_->host_goal_positions();
    torch::Tensor goal_norms = goal_pos_host.norm(2, 1);
    float gd_min = goal_norms.min().item<float>();
    float gd_mean = goal_norms.mean().item<float>();
    total_goal_distance += static_cast<double>(gd_mean);
    if (gd_min < min_goal_distance) {
      min_goal_distance = static_cast<double>(gd_min);
    }

    torch::Tensor terminal_obs_host = collector_->host_terminal_observations();

    const auto learner_step_count = static_cast<std::int64_t>(learner_active_host.sum().item<float>());
    total_reward += extrinsic_rewards_host.sum().item<double>();
    total_gameplay_reward += gameplay_r_host.sum().item<double>();
    total_mechanic_reward += mechanic_r_host.sum().item<double>();
    total_steps += extrinsic_rewards_host.numel();
    total_learner_steps += learner_step_count;

    std::unordered_map<std::string, torch::Tensor> all_values;
    all_values["extrinsic"] = sampled_value_cpu;

    std::unordered_map<std::string, torch::Tensor> all_rewards;
    all_rewards["extrinsic"] = extrinsic_rewards_host;

    dest.append(
        step,
        normalized_obs.to(torch::kCPU),
        episode_starts_host.to(torch::kBool),
        action_masks_host,
        learner_active_host,
        action_indices_cpu,
        action_log_probs.to(torch::kCPU),
        all_values,
        all_rewards,
        dones_host,
        truncated_host,
        bootstrap_truncated_host,
        goal_pos_host,
        terminal_labels,
        terminal_obs_host);

    local_collected_steps += learner_step_count;
    }
  }
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "bootstrap_forward");
    torch::NoGradGuard no_grad;
    torch::Tensor final_raw_obs = collector_->host_observations().to(device_, use_pinned_host_buffers_);
    torch::Tensor final_normalized = actor_normalizer_.normalize(final_raw_obs);
    torch::Tensor final_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    const torch::Tensor final_goal_values = policy_goal_values_like(final_normalized, config_.goal_critic.goal_dim);
    ActorStepOutput final_output = rollout_actor->forward_step(final_normalized, final_goal_values);

    std::unordered_map<std::string, torch::Tensor> bootstrap_values;
    bootstrap_values["extrinsic"] = final_output.value_win_logits.squeeze(-1).to(torch::kCPU);
    dest.set_final_values(bootstrap_values);
    }
  }

  const double collection_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

  if (total_learner_steps > 0) {
    metrics.total_reward_mean = total_reward / static_cast<double>(total_learner_steps);
    metrics.gameplay_reward_mean = total_gameplay_reward / static_cast<double>(total_learner_steps);
    metrics.mechanic_reward_mean = total_mechanic_reward / static_cast<double>(total_learner_steps);
    metrics.mean_goal_distance = total_goal_distance / static_cast<double>(std::max(dest.rollout_length(), 1));
  }
  metrics.min_goal_distance = min_goal_distance;
  metrics.goals_scored = total_goals_scored;
  metrics.goals_conceded = total_goals_conceded;
  metrics.rollout_steps = dest.rollout_length();
  metrics.completed_episodes = completed_episodes;
  metrics.scored_episodes = scored_episodes;
  metrics.touch_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(touched_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  if (total_ball_proximity_denom > 0) {
    metrics.ball_proximity_rate = static_cast<double>(total_ball_proximity_steps) / static_cast<double>(total_ball_proximity_denom);
  }
  if (accumulated_value_count > 0) {
    metrics.sampled_value_win_mean = accumulated_sampled_value
        / static_cast<double>(accumulated_value_count);
  }

  metrics.obs_build_seconds = collector_timings.obs_build_seconds;
  metrics.mask_build_seconds = collector_timings.mask_build_seconds;
  metrics.env_step_seconds = collector_timings.env_step_seconds;
  metrics.done_reset_seconds = collector_timings.done_reset_seconds;
  metrics.collection_agent_steps_per_second =
      local_collected_steps > 0 ? static_cast<double>(local_collected_steps) / collection_seconds : 0.0;

  *collected_agent_steps = local_collected_steps;
}

CheckpointMetadata APPOTrainer::make_checkpoint_metadata(std::int64_t global_step, int update_index, const std::string& wandb_run_id) const {
  nlohmann::json extra = nlohmann::json::object();
  if (!wandb_run_id.empty()) {
    extra["wandb_run_id"] = wandb_run_id;
  }
  if (self_play_manager_ && self_play_manager_->enabled()) {
    extra["self_play_ratings"] = self_play_manager_->current_ratings();
    const std::string rng = self_play_manager_->rng_state();
    if (!rng.empty()) {
      extra["self_play_rng_state"] = rng;
    }
  }
  extra["curriculum_stage"] = curriculum_.state().stage_index;
  extra["curriculum_agent_steps"] = curriculum_.state().agent_steps_in_stage;
  extra["curriculum_promotion_counter"] = curriculum_.state().promotion_counter;
  extra["curriculum_demotion_counter"] = curriculum_.state().demotion_counter;
  extra["curriculum_current_mode"] = curriculum_.state().current_mode;
  nlohmann::json touch_rates_json = nlohmann::json::array();
  for (double v : curriculum_.state().touch_rates) touch_rates_json.push_back(v);
  extra["curriculum_touch_rates"] = touch_rates_json;
  nlohmann::json scored_rates_json = nlohmann::json::array();
  for (double v : curriculum_.state().scored_rates) scored_rates_json.push_back(v);
  extra["curriculum_scored_rates"] = scored_rates_json;
  {
    nlohmann::json recent_json = nlohmann::json::array();
    for (double v : recent_scored_rates_) recent_json.push_back(v);
    extra["recent_scored_rates"] = recent_json;
  }
  return CheckpointMetadata{
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_hash(config_.action_table),
      .architecture_name = "swa_transformer_goal_appo",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
      .critic_heads = {"extrinsic"},
      .extra = std::move(extra),
  };
}

void APPOTrainer::save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index, const std::string& wandb_run_id) const {
  PULSAR_TRACE_SCOPE_CAT("trainer", "checkpoint_save");
  synchronize_cuda_if_needed(device_, "checkpoint save start");
  const std::filesystem::path staging = make_checkpoint_staging_directory(directory);
  remove_checkpoint_directory(staging);
  try {
    std::filesystem::create_directories(staging);
    save_experiment_config(config_, (staging / "config.json").string());
    save_checkpoint_metadata(make_checkpoint_metadata(global_step, update_index, wandb_run_id), (staging / "metadata.json").string());
    save_training_state(staging / "state.pt");
    commit_checkpoint_directory(staging, directory);
    synchronize_cuda_if_needed(device_, "checkpoint save end");
  } catch (...) {
    remove_checkpoint_directory(staging);
    throw;
  }
}

void APPOTrainer::save_training_state(const std::filesystem::path& path) const {
  torch::NoGradGuard no_grad;
  PPOActor actor_cpu = clone_ppo_actor(actor_, torch::Device(torch::kCPU));
  ObservationNormalizer normalizer_cpu = actor_normalizer_.clone();
  normalizer_cpu.to(torch::Device(torch::kCPU));

  torch::serialize::OutputArchive archive;
  actor_cpu->save(archive);
  normalizer_cpu.save(archive);
  actor_optimizer_.save(archive);
  archive.save_to(path.string());
}

void APPOTrainer::load_training_state(const std::filesystem::path& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), device_);
  actor_->load(archive);
  actor_normalizer_.load(archive);
  actor_optimizer_.load(archive);
  actor_->to(device_);
  actor_normalizer_.to(device_);
}

void APPOTrainer::prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const {
  const int max_checkpoints = config_.ppo.max_rolling_checkpoints;
  if (max_checkpoints <= 0) {
    return;
  }
  std::error_code ec;
  if (!std::filesystem::exists(checkpoint_dir, ec)) {
    return;
  }
  std::vector<std::pair<int, std::filesystem::path>> updates;
  for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir, ec)) {
    if (ec) break;
    if (!entry.is_directory(ec)) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.rfind("update_", 0) != 0) {
      continue;
    }
    const std::string suffix = name.substr(7);
    if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](char ch) { return ch >= '0' && ch <= '9'; })) {
      continue;
    }
    try {
      updates.emplace_back(std::stoi(suffix), entry.path());
    } catch (...) {
    }
  }
  std::sort(updates.begin(), updates.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
  for (std::size_t i = static_cast<std::size_t>(max_checkpoints); i < updates.size(); ++i) {
    remove_checkpoint_directory(updates[i].second);
  }
}

void APPOTrainer::train(int updates, const std::string& checkpoint_dir, const std::string& config_path) {
  const bool train_forever = updates <= 0;
  std::cout << "train_start curriculum_enabled=" << (curriculum_.enabled() ? 1 : 0)
            << " stages=" << config_.curriculum.stages.size()
            << '\n';
  std::filesystem::create_directories(checkpoint_dir);
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "dappo_train");
  std::int64_t global_step = resumed_global_step_;

  if (curriculum_.enabled()) {
    apply_curriculum_to_collectors();
    apply_curriculum_lr();
  }

  TrainerMetrics coll_metrics{};
  std::int64_t coll_steps = 0;
  collect_rollout(rollout_, coll_metrics, &coll_steps, actor_snapshot_);
  global_step += coll_steps;

  for (int index = 0; train_forever || index < updates; ++index) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "train_iteration");
    const auto iter_start = std::chrono::steady_clock::now();
    const int update_index = static_cast<int>(resumed_update_index_) + index + 1;

    TrainerMetrics next_coll_metrics{};
    std::int64_t next_coll_steps = 0;
    const bool has_next = train_forever || index + 1 < updates;

    TrainerMetrics train_metrics = update_actor(rollout_);

    {
      double scored = static_cast<double>(coll_metrics.scored_episodes);
      double completed = static_cast<double>(std::max(coll_metrics.completed_episodes, static_cast<int64_t>(1)));
      double scored_rate = scored / completed;
      recent_scored_rates_.push_back(scored_rate);
      coll_metrics.scored_episode_rate = scored_rate;
      if (static_cast<int>(recent_scored_rates_.size()) > kRecentScoredRateWindow) {
        recent_scored_rates_.pop_front();
      }
    }
    if (curriculum_.enabled()) {
      if (curriculum_.check_promotion(
          coll_metrics.touch_episode_rate, coll_metrics.scored_episode_rate, coll_steps)) {
        if (curriculum_.mode_changed()) {
          std::cout << "curriculum_mode_change new_mode=" << curriculum_.current_mode()
                    << " (collector rebuild not yet implemented — restart required)\n";
        }
        apply_curriculum_to_collectors();
        apply_curriculum_lr();
      } else {
        bool demoted = curriculum_.check_demotion(coll_metrics.scored_episode_rate);
        if (demoted) {
          if (curriculum_.mode_changed()) {
            std::cout << "curriculum_mode_change new_mode=" << curriculum_.current_mode()
                      << " (collector rebuild not yet implemented — restart required)\n";
          }
          apply_curriculum_to_collectors();
          apply_curriculum_lr();
        }
      }
    }

    if (self_play_manager_) {
      const SelfPlayMetrics self_play_metrics =
          self_play_manager_->on_update(actor_, actor_normalizer_, global_step, update_index);
      coll_metrics.self_play_eval_seconds = self_play_metrics.eval_seconds;
      coll_metrics.elo_ratings = self_play_metrics.ratings;
    }

    if (update_index % config_.es_lora.es_interval == 0) {
      run_es_lora_update(update_index, coll_metrics);
    }

    if (config_.ppo.plasticity && update_index % config_.ppo.plasticity_interval == 0) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "plasticity");
      shrink_perturb_parameters(*actor_, config_.ppo.plasticity_shrink, config_.ppo.plasticity_noise);
      actor_->to(device_);
      for (auto& p : actor_->parameters()) {
        if (!p.requires_grad() || p.dim() < 2) continue;
        actor_optimizer_.state().erase(p.unsafeGetTensorImpl());
      }
    }

    synchronize_cuda_if_needed(device_, "snapshot clone");
    actor_snapshot_ = clone_ppo_actor(actor_, device_);
    actor_snapshot_->eval();

    if (has_next) {
      collect_rollout(rollout_B_, next_coll_metrics, &next_coll_steps, actor_snapshot_);
    }

    global_step += next_coll_steps;

    coll_metrics.policy_loss = train_metrics.policy_loss;
    coll_metrics.value_loss = train_metrics.value_loss;
    coll_metrics.entropy = train_metrics.entropy;
    coll_metrics.effective_entropy_coef = train_metrics.effective_entropy_coef;
    coll_metrics.grad_norm = train_metrics.grad_norm;
    coll_metrics.update_seconds = train_metrics.update_seconds;
    coll_metrics.forward_backward_seconds = train_metrics.forward_backward_seconds;
    coll_metrics.optimizer_step_seconds = train_metrics.optimizer_step_seconds;
    coll_metrics.goal_critic_loss = train_metrics.goal_critic_loss;
    coll_metrics.mean_goal_score = train_metrics.mean_goal_score;
    coll_metrics.mean_sampled_goal_distance = train_metrics.mean_sampled_goal_distance;
    coll_metrics.update_agent_steps_per_second =
        next_coll_steps > 0 ? static_cast<double>(next_coll_steps) / std::max(train_metrics.update_seconds, 1.0e-9) : 0.0;

    coll_metrics.overall_agent_steps_per_second =
        next_coll_steps > 0
            ? static_cast<double>(next_coll_steps) /
                  std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count(), 1.0e-9)
            : 0.0;

    append_metrics_line(checkpoint_dir, update_index, global_step, coll_metrics);
    std::cout << "update=" << update_index
              << " global_step=" << global_step
              << " policy_loss=" << coll_metrics.policy_loss
              << " value_loss=" << coll_metrics.value_loss
              << " entropy=" << coll_metrics.entropy
              << " grad_norm=" << coll_metrics.grad_norm
              << " total_reward=" << coll_metrics.total_reward_mean
              << " gameplay_reward=" << coll_metrics.gameplay_reward_mean
              << " mechanic_reward=" << coll_metrics.mechanic_reward_mean
              << " rollout_steps=" << coll_metrics.rollout_steps
              << " completed_eps=" << coll_metrics.completed_episodes
              << " scored_eps=" << coll_metrics.scored_episodes
              << " touch_rate=" << coll_metrics.touch_episode_rate
              << " sampled_goal_dist=" << coll_metrics.mean_sampled_goal_distance
              << " mean_goal_dist=" << coll_metrics.mean_goal_distance
              << " ball_prox=" << coll_metrics.ball_proximity_rate
              << " goals=" << coll_metrics.goals_scored << "/" << coll_metrics.goals_conceded
              << " es_fitness=" << coll_metrics.es_fitness_mean
              << " curriculum=" << curriculum_.state().stage_index
              << " cur_steps=" << curriculum_.state().agent_steps_in_stage
              << " cur_promo=" << curriculum_.state().promotion_counter
              << '\n';
    if (wandb.enabled()) {
      nlohmann::json payload{
          {"_step", global_step},
          {"update", update_index},
          {"global_step", global_step},
          {"policy_loss", coll_metrics.policy_loss},
          {"value_loss", coll_metrics.value_loss},
          {"entropy", coll_metrics.entropy},
      {"total_reward_mean", coll_metrics.total_reward_mean},
      {"gameplay_reward_mean", coll_metrics.gameplay_reward_mean},
      {"mechanic_reward_mean", coll_metrics.mechanic_reward_mean},
          {"sampled_value_win_mean", coll_metrics.sampled_value_win_mean},
          {"rollout_steps", coll_metrics.rollout_steps},
          {"completed_episodes", coll_metrics.completed_episodes},
      {"scored_episodes", coll_metrics.scored_episodes},
      {"touch_episode_rate", coll_metrics.touch_episode_rate},
          {"goal_critic_loss", coll_metrics.goal_critic_loss},
          {"mean_goal_score", coll_metrics.mean_goal_score},
          {"mean_sampled_goal_distance", coll_metrics.mean_sampled_goal_distance},
          {"mean_goal_distance", coll_metrics.mean_goal_distance},
          {"min_goal_distance", coll_metrics.min_goal_distance},
          {"ball_proximity_rate", coll_metrics.ball_proximity_rate},
          {"goals_scored", coll_metrics.goals_scored},
          {"goals_conceded", coll_metrics.goals_conceded},
      {"scored_episode_rate", coll_metrics.scored_episode_rate},
          {"effective_entropy_coef", coll_metrics.effective_entropy_coef},
          {"curriculum_stage", curriculum_.state().stage_index},
          {"curriculum_agent_steps", curriculum_.state().agent_steps_in_stage},
          {"curriculum_promotion_counter", curriculum_.state().promotion_counter},
      };
      if (update_index % config_.es_lora.es_interval == 0) {
        payload["es_fitness_mean"] = coll_metrics.es_fitness_mean;
        payload["es_fitness_std"] = coll_metrics.es_fitness_std;
        payload["es_fitness_best"] = coll_metrics.es_fitness_best;
        payload["es_winrate_mean"] = coll_metrics.es_winrate_mean;
        payload["es_kl_mean"] = coll_metrics.es_kl_mean;
        payload["es_update_norm"] = coll_metrics.es_update_norm;
        payload["es_lora_a_norm"] = coll_metrics.es_lora_a_norm;
        payload["es_lora_b_norm"] = coll_metrics.es_lora_b_norm;
      }
      for (const auto& [mode, rating] : coll_metrics.elo_ratings) {
        payload["elo_" + mode] = rating;
      }
      wandb.log(payload);
    }
    if (config_.ppo.checkpoint_interval > 0 && update_index % config_.ppo.checkpoint_interval == 0) {
      save_checkpoint(std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(update_index)), global_step, update_index, wandb.run_id());
      prune_old_checkpoints(checkpoint_dir);
    }

    if (has_next) {
      std::swap(rollout_, rollout_B_);
      coll_metrics = std::move(next_coll_metrics);
    }
    coll_steps = next_coll_steps;
  }
  save_checkpoint(std::filesystem::path(checkpoint_dir) / "final", global_step, static_cast<int>(resumed_update_index_) + updates, wandb.run_id());
  wandb.finish();
}

}  // namespace pulsar

#endif
