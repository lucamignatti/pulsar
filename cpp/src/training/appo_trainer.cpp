#include "pulsar/training/appo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
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
      config.model.encoder_dim,
      torch::Device(torch::kCPU));
}

void require_finite(const torch::Tensor& tensor, const std::string& name) {
  if (tensor.defined() && !torch::isfinite(tensor).all().item<bool>()) {
    throw std::runtime_error("Non-finite tensor: " + name);
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
      {"policy_loss", metrics.policy_loss},
      {"value_loss", metrics.value_loss},
      {"entropy", metrics.entropy},
      {"grad_norm", metrics.grad_norm},
      {"adaptive_epsilon", metrics.adaptive_epsilon},
      {"critic_variance", metrics.critic_variance},
      {"mean_confidence_weight", metrics.mean_confidence_weight},
      {"sparse_reward_mean", metrics.sparse_reward_mean},
      {"sampled_value_win_mean", metrics.sampled_value_win_mean},
      {"value_win_entropy", metrics.value_win_entropy},
      {"goal_critic_loss", metrics.goal_critic_loss},
      {"goal_actor_loss", metrics.goal_actor_loss},
      {"mean_predicted_goal_value", metrics.mean_predicted_goal_value},
      {"mean_actual_goal_occupancy", metrics.mean_actual_goal_occupancy},
      {"mean_goal_distance", metrics.mean_goal_distance},
      {"min_goal_distance", metrics.min_goal_distance},
      {"goal_actor_loss_ratio", metrics.goal_actor_loss_ratio},
      {"goal_occupancy_correlation", metrics.goal_occupancy_correlation},
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
      {"es_goal_pressure_mean", metrics.es_goal_pressure_mean},
      {"es_kl_mean", metrics.es_kl_mean},
      {"es_update_norm", metrics.es_update_norm},
      {"es_lora_a_norm", metrics.es_lora_a_norm},
      {"es_lora_b_norm", metrics.es_lora_b_norm},
      {"es_seconds", metrics.es_seconds},
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
      actor_(PPOActor(config_.model, config_.goal_critic)),
      actor_normalizer_(config_.model.observation_dim),
      actor_optimizer_(actor_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate)),
      rollout_(make_rollout_storage(
          config_,
          static_cast<int>(collector_->total_agents()),
          collector_->action_dim())),
      device_(resolve_runtime_device(config_.ppo.device)),
      run_output_root_(std::move(run_output_root)),
      log_initialization_(log_initialization) {
  validate_experiment_config(config_);
  if (!collector_) {
    throw std::invalid_argument("APPOTrainer requires a collector.");
  }
  total_agents_ = collector_->total_agents();
  seed_everything(config_.env.seed);
  collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  opponent_collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  configure_cuda_runtime(device_);
  use_pinned_host_buffers_ = device_.is_cuda();
  actor_->to(device_);
  actor_normalizer_.to(device_);

  maybe_initialize_from_checkpoint();

  if (self_play_manager_ && self_play_manager_->enabled()) {
    collector_->set_self_play_assignment_fn(
        [this](std::size_t env_idx, std::uint64_t seed) {
          return self_play_manager_->sample_assignment(env_idx, seed);
        });
  }
}

APPOTrainer::~APPOTrainer() {
  synchronize_cuda_if_needed(device_, "trainer shutdown");
}

torch::Tensor APPOTrainer::map_outcome_labels_to_rewards(const torch::Tensor& labels) const {
  torch::Tensor rewards = torch::zeros_like(labels, torch::TensorOptions().dtype(torch::kFloat32));
  rewards.masked_fill_(labels == 0, config_.outcome.score);
  rewards.masked_fill_(labels == 1, config_.outcome.concede);
  rewards.masked_fill_(labels == 2, config_.outcome.neutral);
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
    std::cout << "initialized_from_checkpoint=" << base.string() << '\n';
  }
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
  double accumulated_goal_critic_loss = 0.0;
  double accumulated_goal_actor_loss = 0.0;
  double accumulated_predicted_goal_value = 0.0;
  double accumulated_actual_goal_occupancy = 0.0;
  std::vector<double> predicted_goal_values;
  std::vector<double> actual_goal_occupancies;
  std::vector<double> correlation_weights;

  const auto& all_values = rollout_.all_values();
  const auto& all_rewards = rollout_.all_rewards();

  torch::Tensor sparse_advantages = compute_gae(
      all_values.at("extrinsic"), all_rewards.at("extrinsic"),
      rollout_.dones, config_.ppo.gamma, config_.ppo.gae_lambda,
      rollout_.final_values().count("extrinsic") ? rollout_.final_values().at("extrinsic") : torch::Tensor{});

  torch::Tensor active_mask = rollout_.learner_active > 0.5F;
  torch::Tensor normalized_advantages = normalize_advantage(sparse_advantages, active_mask);
  torch::Tensor sparse_returns = sparse_advantages + all_values.at("extrinsic").detach();

  const torch::Tensor atom_support_win = actor_->value_win_support().to(device_);

  for (int epoch = 0; epoch < config_.ppo.update_epochs; ++epoch) {
    const torch::Tensor perm = torch::randperm(total_agents, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    for (int agent_offset = 0; agent_offset < total_agents; agent_offset += agents_per_batch) {
      const int count = std::min(agents_per_batch, total_agents - agent_offset);
      const torch::Tensor agent_indices = perm.narrow(0, agent_offset, count);
      ContinuumState state = state_to_device(rollout_.initial_state_for_agents(agent_indices), device_);

      double total_active_samples_agent = 0.0;

      for (int seq_start = 0; seq_start < rollout_.rollout_length(); seq_start += seq_len) {
        const int chunk_start = seq_start;
        const int chunk_end = std::min(rollout_.rollout_length(), chunk_start + seq_len);
        const int chunk_steps = chunk_end - chunk_start;
        const int burn = seq_start == 0 ? std::min(std::max(0, config_.ppo.burn_in), chunk_steps) : 0;
        const int loss_start = chunk_start + burn;
        const int loss_steps = chunk_steps - burn;
        if (loss_steps <= 0) {
          continue;
        }
        total_active_samples_agent += rollout_.learner_active
            .narrow(0, loss_start, loss_steps)
            .index_select(1, agent_indices)
            .sum()
            .item<double>();
      }
      if (total_active_samples_agent <= 0.0) {
        continue;
      }

      actor_optimizer_.zero_grad();

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
        torch::Tensor features = output.features.narrow(0, burn, loss_steps);

        const torch::Tensor action_masks =
            rollout_.action_masks.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_).to(torch::kBool);
        const torch::Tensor learner_active =
            rollout_.learner_active.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor old_actions =
            rollout_.actions.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor old_log_probs =
            rollout_.action_log_probs.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
        const torch::Tensor chunk_advantages =
            normalized_advantages.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);

        const auto samples = loss_steps * count;
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
        torch::Tensor confidence_weights = torch::ones({active_advantages.size(0)}, active_advantages.options());

        {
          torch::Tensor value_win_chunk = output.value_win_logits.narrow(0, burn, loss_steps);
          const int64_t win_atoms = atom_support_win.size(0);
          torch::Tensor flat_value_logits = value_win_chunk.reshape({samples, win_atoms});
          torch::Tensor active_value_logits = flat_value_logits.index({flat_active});
          const torch::Tensor critic_variance = compute_distribution_variance(active_value_logits, atom_support_win);
          accumulated_variance += critic_variance.mean().item<double>() * static_cast<double>(active_logits.size(0));

          if (config_.ppo.use_adaptive_epsilon) {
            epsilon = compute_adaptive_epsilon_tensor(
                critic_variance,
                config_.ppo.clip_range,
                config_.ppo.adaptive_epsilon_beta,
                config_.ppo.epsilon_min,
                config_.ppo.epsilon_max);
          }
          if (config_.ppo.use_confidence_weighting) {
            confidence_weights = compute_confidence_weights(
                active_value_logits,
                atom_support_win,
                config_.ppo.confidence_weight_type,
                config_.ppo.confidence_weight_delta,
                config_.ppo.normalize_confidence_weights);
          }
          accumulated_confidence += confidence_weights.mean().item<double>() * static_cast<double>(active_logits.size(0));
        }
        accumulated_epsilon += static_cast<double>(epsilon.mean().item<float>()) * static_cast<double>(active_logits.size(0));

        torch::Tensor policy_loss =
            clipped_ppo_policy_loss(current_log_probs, active_old_log_probs, active_advantages, epsilon);
        policy_loss = (policy_loss * confidence_weights).mean();

        const torch::Tensor entropy = masked_action_entropy(active_logits, active_masks).mean();

        torch::Tensor chunk_returns =
            sparse_returns.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_).reshape({samples});
        torch::Tensor active_returns = chunk_returns.index({flat_active});

        torch::Tensor value_win_chunk = output.value_win_logits.narrow(0, burn, loss_steps);
        const int64_t win_atoms = atom_support_win.size(0);
        torch::Tensor flat_value_win_logits = value_win_chunk.reshape({samples, win_atoms});
        torch::Tensor active_value_win_logits = flat_value_win_logits.index({flat_active});

        const float v_min = atom_support_win[0].item<float>();
        const float v_max = atom_support_win[-1].item<float>();
        const int num_atoms = static_cast<int>(atom_support_win.size(0));
        torch::Tensor value_loss = distributional_value_loss(
            active_value_win_logits, active_returns, v_min, v_max, num_atoms);

        torch::Tensor goal_critic_loss = torch::zeros({}, active_advantages.options());
        torch::Tensor goal_actor_loss = torch::zeros({}, active_advantages.options());
        double chunk_goal_value = 0.0;

        {
          torch::Tensor chunk_goal_dist =
              rollout_.goal_distances.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_);
          torch::Tensor flat_goal_dist = chunk_goal_dist.reshape({samples});
          torch::Tensor active_goal_dist = flat_goal_dist.index({flat_active});

          torch::Tensor goal_occupancy = compute_finite_horizon_goal_occupancy(
              chunk_goal_dist.unsqueeze(-1).reshape({loss_steps, count}),
              rollout_.dones.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_),
              config_.goal_critic.gamma_g,
              config_.goal_mapping.goal,
              config_.goal_mapping.kernel_sigma,
              config_.goal_critic.horizon_H);

          torch::Tensor flat_goal_occ = goal_occupancy.reshape({samples});
          torch::Tensor active_goal_occ = flat_goal_occ.index({flat_active});

          torch::Tensor goal_critic_logits = actor_->goal_critic()->forward(
              active_features, active_actions,
              torch::full({active_actions.size(0)}, config_.goal_mapping.goal, active_actions.options()));

          const torch::Tensor goal_support = actor_->goal_critic_support().to(device_);
          const float g_v_min = goal_support[0].item<float>();
          const float g_v_max = goal_support[-1].item<float>();
          const int g_num_atoms = static_cast<int>(goal_support.size(0));
          goal_critic_loss = distributional_value_loss(
              goal_critic_logits, active_goal_occ, g_v_min, g_v_max, g_num_atoms);

          goal_actor_loss = compute_goal_actor_loss_discrete(
              active_logits, active_masks, goal_critic_logits.detach(), goal_support, active_actions);

          chunk_goal_value = compute_mean_value(goal_critic_logits, goal_support).mean().item<double>();

          double chunk_actual_occ = active_goal_occ.mean().item<double>();
          accumulated_actual_goal_occupancy += chunk_actual_occ * static_cast<double>(active_logits.size(0));
          predicted_goal_values.push_back(chunk_goal_value);
          actual_goal_occupancies.push_back(chunk_actual_occ);
          correlation_weights.push_back(static_cast<double>(active_logits.size(0)));
        }

        accumulated_goal_critic_loss += goal_critic_loss.item<double>() * static_cast<double>(active_logits.size(0));
        accumulated_goal_actor_loss += goal_actor_loss.item<double>() * static_cast<double>(active_logits.size(0));
        accumulated_predicted_goal_value += chunk_goal_value * static_cast<double>(active_logits.size(0));

        const torch::Tensor loss =
            policy_loss
            + config_.ppo.value_coef * value_loss
            + config_.goal_critic.lambda_Zg * goal_critic_loss
            + config_.actor_goal.lambda_g * goal_actor_loss
            - config_.ppo.entropy_coef * entropy;

        metrics.forward_backward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - forward_start).count();

        require_finite(loss, "loss");
        require_finite(policy_loss, "policy_loss");
        require_finite(value_loss, "value_loss");
        require_finite(entropy, "entropy");
        require_finite(goal_critic_loss, "goal_critic_loss");
        require_finite(goal_actor_loss, "goal_actor_loss");

        const auto active_samples = active_logits.size(0);
        const torch::Tensor weighted_loss = loss * (static_cast<double>(active_samples) / total_active_samples_agent);
        weighted_loss.backward();

        metrics.policy_loss += policy_loss.item<double>() * static_cast<double>(active_samples);
        metrics.value_loss += value_loss.item<double>() * static_cast<double>(active_samples);
        metrics.entropy += entropy.item<double>() * static_cast<double>(active_samples);
        metric_steps += active_samples;
      }

      const auto optim_start = std::chrono::steady_clock::now();
      const auto grad_norm_value = torch::nn::utils::clip_grad_norm_(actor_->parameters(), config_.ppo.max_grad_norm);
      double grad_norm = static_cast<double>(grad_norm_value);
      actor_optimizer_.step();
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
    metrics.adaptive_epsilon = accumulated_epsilon / static_cast<double>(metric_steps);
    metrics.critic_variance = accumulated_variance / static_cast<double>(metric_steps);
    metrics.mean_confidence_weight = accumulated_confidence / static_cast<double>(metric_steps);
    metrics.goal_critic_loss = accumulated_goal_critic_loss / static_cast<double>(metric_steps);
    metrics.goal_actor_loss = accumulated_goal_actor_loss / static_cast<double>(metric_steps);
    metrics.mean_predicted_goal_value = accumulated_predicted_goal_value / static_cast<double>(metric_steps);
    metrics.mean_actual_goal_occupancy = accumulated_actual_goal_occupancy / static_cast<double>(metric_steps);
    metrics.goal_actor_loss_ratio = std::abs(metrics.goal_actor_loss * static_cast<double>(config_.actor_goal.lambda_g))
        / std::max(std::abs(metrics.policy_loss), 1.0e-8);

    if (!predicted_goal_values.empty() && predicted_goal_values.size() == actual_goal_occupancies.size()) {
      auto pred_t = torch::tensor(predicted_goal_values, torch::kFloat32);
      auto act_t = torch::tensor(actual_goal_occupancies, torch::kFloat32);
      auto weights_t = torch::tensor(correlation_weights, torch::kFloat32);
      metrics.goal_occupancy_correlation = static_cast<double>(compute_goal_value_correlation(pred_t, act_t, weights_t));
    }
  }
  metrics.update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  return metrics;
}

APPOTrainer::ESPopulationFitness APPOTrainer::evaluate_es_population(
    const torch::Tensor& A_stack,
    const torch::Tensor& B_stack,
    int update_index) {
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
  result.goal_pressure.assign(static_cast<std::size_t>(pop), 0.0F);
  result.kl.assign(static_cast<std::size_t>(pop), 0.0F);

  std::vector<int> episode_counts(static_cast<std::size_t>(pop), 0);
  std::vector<int> win_counts(static_cast<std::size_t>(pop), 0);

  torch::Tensor goal_sum = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor goal_count = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor kl_sum = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor kl_count = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  const torch::Tensor goal_support = actor_->goal_critic_support().to(device_);

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

  for (int ep = 0; ep < es_cfg.eval_episodes_per_member; ++ep) {
    auto eval_collector = make_es_eval_collector(
        config_, total_envs, eval_envs, update_index, ep, use_pinned_host_buffers_);
    ContinuumState eval_state = actor_->initial_state(
        static_cast<std::int64_t>(eval_collector->total_agents()), device_);

    for (int step = 0; step < es_cfg.eval_rollout_length; ++step) {
      torch::Tensor raw_obs = eval_collector->host_observations().to(device_, use_pinned_host_buffers_);
      torch::Tensor episode_starts = eval_collector->host_episode_starts().to(device_, use_pinned_host_buffers_);
      torch::Tensor action_masks = eval_collector->host_action_masks().to(device_, use_pinned_host_buffers_).to(torch::kBool);
      torch::Tensor normalized_obs = actor_normalizer_.normalize(raw_obs);

      ActorStepOutput output = actor_->forward_step(normalized_obs, std::move(eval_state), episode_starts);
      eval_state = std::move(output.state);
      torch::Tensor perturbed_logits = actor_->policy_eggroll_logits(
          output.features, A_stack, B_stack, es_cfg.sigma_ES);

      torch::Tensor base_actions = sample_masked_actions(output.policy_logits, action_masks, true, nullptr);
      torch::Tensor perturbed_actions = sample_masked_actions(perturbed_logits, action_masks, true, nullptr);
      torch::Tensor actions = torch::where(controlled_mask, perturbed_actions, base_actions);

      torch::Tensor goal_logits = actor_->goal_critic()->forward(
          output.features,
          perturbed_actions,
          torch::full({perturbed_actions.size(0)}, config_.goal_mapping.goal, perturbed_actions.options()));
      torch::Tensor goal_values = compute_mean_value(goal_logits, goal_support).view({pop, member_agents});
      goal_sum += (goal_values * controlled_float).sum(1);
      goal_count += controlled_float.sum(1);

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

  torch::Tensor goal_mean = (goal_sum / goal_count.clamp_min(1.0F)).to(torch::kCPU);
  torch::Tensor kl_mean = (kl_sum / kl_count.clamp_min(1.0F)).to(torch::kCPU);
  const auto* goal_ptr = goal_mean.data_ptr<float>();
  const auto* kl_ptr = kl_mean.data_ptr<float>();
  for (int i = 0; i < pop; ++i) {
    const int denom = std::max(episode_counts[static_cast<std::size_t>(i)], 1);
    result.winrate[static_cast<std::size_t>(i)] =
        static_cast<float>(win_counts[static_cast<std::size_t>(i)]) / static_cast<float>(denom);
    result.goal_pressure[static_cast<std::size_t>(i)] = goal_ptr[i];
    result.kl[static_cast<std::size_t>(i)] = kl_ptr[i];
    result.fitness[static_cast<std::size_t>(i)] =
        result.winrate[static_cast<std::size_t>(i)]
        + es_cfg.alpha_g * result.goal_pressure[static_cast<std::size_t>(i)]
        - es_cfg.beta_KL * result.kl[static_cast<std::size_t>(i)];
  }
  return result;
}

void APPOTrainer::run_es_lora_update(int update_index, TrainerMetrics& metrics) {
  const auto es_start = std::chrono::steady_clock::now();
  const auto& es_cfg = config_.es_lora;
  const int pop = es_cfg.population_size;
  const int rank = es_cfg.rank;
  const int in_features = actor_->policy_lora()->in_features();
  const int out_features = actor_->policy_lora()->out_features();

  torch::Tensor A_stack = torch::randn(
      {pop, rank, in_features},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor B_stack = torch::randn(
      {pop, out_features, rank},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

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

  torch::Tensor delta_weight = torch::zeros(
      {out_features, in_features},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  for (uint64_t i = 0; i < total_members; ++i) {
    delta_weight.add_(
        torch::matmul(B_stack[static_cast<long>(i)], A_stack[static_cast<long>(i)]),
        normalized_f[i]);
  }
  delta_weight.div_(static_cast<float>(total_members) * es_cfg.sigma_ES);
  delta_weight.mul_(es_cfg.eta_ES);

  double update_norm = static_cast<double>(delta_weight.norm().item<float>());

  if (es_cfg.update_norm_clip) {
    const double param_norm = static_cast<double>(actor_->policy_lora()->base->weight.norm().item<float>());
    double clip_val = 0.1 * param_norm;
    if (update_norm > clip_val) {
      double scale = clip_val / update_norm;
      delta_weight.mul_(static_cast<float>(scale));
      update_norm = clip_val;
    }
  }

  actor_->apply_policy_eggroll_update(delta_weight);

  float best_fitness = *std::max_element(fitnesses.begin(), fitnesses.end());

  metrics.es_fitness_mean = mu;
  metrics.es_fitness_std = sigma;
  metrics.es_fitness_best = static_cast<double>(best_fitness);
  metrics.es_update_norm = update_norm;
  for (uint64_t i = 0; i < total_members; ++i) {
    metrics.es_winrate_mean += population.winrate[i];
    metrics.es_goal_pressure_mean += population.goal_pressure[i];
    metrics.es_kl_mean += population.kl[i];
  }
  metrics.es_winrate_mean /= static_cast<double>(total_members);
  metrics.es_goal_pressure_mean /= static_cast<double>(total_members);
  metrics.es_kl_mean /= static_cast<double>(total_members);

  auto lora_params = actor_->es_lora_parameters();
  metrics.es_lora_a_norm = static_cast<double>(lora_params[0].norm().item<float>());
  metrics.es_lora_b_norm = static_cast<double>(lora_params[1].norm().item<float>());

  metrics.es_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - es_start).count();
}

TrainerMetrics APPOTrainer::run_update(std::int64_t* global_step, int update_index) {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  CollectorTimings collector_timings{};
  std::int64_t collected_agent_steps = 0;

  const auto collection_start = std::chrono::steady_clock::now();
  rollout_.set_initial_state(collection_state_);
  const torch::Tensor atom_support_win = actor_->value_win_support().to(device_);

  double total_sparse_reward = 0.0;
  int64_t total_steps = 0;
  int64_t total_learner_steps = 0;
  double accumulated_sampled_value = 0.0;
  double accumulated_value_entropy = 0.0;
  int64_t accumulated_value_stat_count = 0;
  double total_goal_distance = 0.0;
  double min_goal_distance = 1.0;
  int64_t total_goals_scored = 0;
  int64_t total_goals_conceded = 0;
  int64_t total_ball_proximity_steps = 0;
  int64_t total_ball_proximity_denom = 0;

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
    if (config_.ppo.synchronize_cuda_timing && device_.is_cuda()) {
      torch::cuda::synchronize();
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

    torch::Tensor sampled_value = compute_mean_value(output.value_win_logits, atom_support_win);
    accumulated_sampled_value += sampled_value.sum().item<double>();
    {
      torch::Tensor entropy_tensor = compute_distribution_entropy(output.value_win_logits);
      accumulated_value_entropy += entropy_tensor.sum().item<double>();
      accumulated_value_stat_count += static_cast<int64_t>(sampled_value.numel());
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
    torch::Tensor terminal_labels = collector_->host_terminal_outcome_labels();
    torch::Tensor extrinsic_rewards = map_outcome_labels_to_rewards(terminal_labels);
    extrinsic_rewards = extrinsic_rewards.to(device_, use_pinned_host_buffers_) * dones;

    torch::Tensor ball_prox_host = collector_->host_ball_proximity();
    total_ball_proximity_steps += ball_prox_host.sum().item<int64_t>();
    total_ball_proximity_denom += ball_prox_host.numel();

    torch::Tensor terminal_labels_cpu = terminal_labels.to(torch::kCPU);
    auto* tl_ptr = terminal_labels_cpu.data_ptr<std::int64_t>();
    torch::Tensor learner_active_cpu_early = learner_active.to(torch::kCPU);
    auto* la_ptr = learner_active_cpu_early.data_ptr<float>();
    for (int64_t i = 0; i < terminal_labels_cpu.numel(); ++i) {
      if (la_ptr[i] > 0.5F && dones.to(torch::kCPU)[i].item<float>() > 0.5F) {
        if (tl_ptr[i] == 0) {
          total_goals_scored++;
        } else if (tl_ptr[i] == 1) {
          total_goals_conceded++;
        }
      }
    }

    torch::Tensor goal_dist_host = collector_->host_goal_distances();
    float gd_min = goal_dist_host.min().item<float>();
    float gd_mean = goal_dist_host.mean().item<float>();
    total_goal_distance += static_cast<double>(gd_mean);
    if (gd_min < min_goal_distance) {
      min_goal_distance = static_cast<double>(gd_min);
    }

    total_sparse_reward += extrinsic_rewards.sum().item<double>();
    total_steps += extrinsic_rewards.numel();
    total_learner_steps += learner_active.to(torch::kCPU).sum().item<int64_t>();

    torch::Tensor episode_starts_cpu = episode_starts.to(torch::kCPU).to(torch::kBool);
    torch::Tensor learner_active_cpu = learner_active.to(torch::kCPU).to(torch::kBool);
    torch::Tensor dones_cpu = dones.to(torch::kCPU).to(torch::kBool);

    std::unordered_map<std::string, torch::Tensor> all_values;
    all_values["extrinsic"] = sampled_value.to(torch::kCPU);

    std::unordered_map<std::string, torch::Tensor> all_rewards;
    all_rewards["extrinsic"] = extrinsic_rewards.to(torch::kCPU);

    rollout_.append(
        step,
        raw_obs_host,
        normalized_obs.to(torch::kCPU),
        output.encoded.to(torch::kCPU),
        episode_starts_cpu,
        action_masks.to(torch::kUInt8).to(torch::kCPU),
        learner_active.to(torch::kCPU),
        actions.to(torch::kCPU),
        action_log_probs.to(torch::kCPU),
        all_values,
        all_rewards,
        dones.to(torch::kCPU),
        goal_dist_host);

    collected_agent_steps += learner_active.sum().item<std::int64_t>();
  }
  rollout_.set_final_observation(collector_->host_observations());

  {
    torch::NoGradGuard no_grad;
    torch::Tensor final_raw_obs = collector_->host_observations().to(device_, use_pinned_host_buffers_);
    torch::Tensor final_normalized = actor_normalizer_.normalize(final_raw_obs);
    torch::Tensor final_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    ContinuumState bootstrap_state = clone_state(collection_state_);
    ActorStepOutput final_output = actor_->forward_step(
        final_normalized, std::move(bootstrap_state), final_starts);

    std::unordered_map<std::string, torch::Tensor> bootstrap_values;
    bootstrap_values["extrinsic"] = compute_mean_value(
        final_output.value_win_logits, atom_support_win).to(torch::kCPU);
    rollout_.set_final_values(bootstrap_values);
    rollout_.set_final_encoded(final_output.encoded.to(torch::kCPU));
  }

  const double collection_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

  if (total_learner_steps > 0) {
    metrics.sparse_reward_mean = total_sparse_reward / static_cast<double>(total_learner_steps);
    metrics.mean_goal_distance = total_goal_distance / static_cast<double>(config_.ppo.rollout_length);
  }
  metrics.min_goal_distance = min_goal_distance;
  metrics.goals_scored = total_goals_scored;
  metrics.goals_conceded = total_goals_conceded;
  if (total_ball_proximity_denom > 0) {
    metrics.ball_proximity_rate = static_cast<double>(total_ball_proximity_steps) / static_cast<double>(total_ball_proximity_denom);
  }
  if (accumulated_value_stat_count > 0) {
    metrics.sampled_value_win_mean = accumulated_sampled_value
        / static_cast<double>(accumulated_value_stat_count);
    metrics.value_win_entropy = accumulated_value_entropy
        / static_cast<double>(accumulated_value_stat_count);
  }

  TrainerMetrics update_metrics = update_actor();
  metrics.policy_loss = update_metrics.policy_loss;
  metrics.value_loss = update_metrics.value_loss;
  metrics.entropy = update_metrics.entropy;
  metrics.grad_norm = update_metrics.grad_norm;
  metrics.update_seconds = update_metrics.update_seconds;
  metrics.forward_backward_seconds = update_metrics.forward_backward_seconds;
  metrics.optimizer_step_seconds = update_metrics.optimizer_step_seconds;
  metrics.adaptive_epsilon = update_metrics.adaptive_epsilon;
  metrics.critic_variance = update_metrics.critic_variance;
  metrics.mean_confidence_weight = update_metrics.mean_confidence_weight;
  metrics.goal_critic_loss = update_metrics.goal_critic_loss;
  metrics.goal_actor_loss = update_metrics.goal_actor_loss;
  metrics.mean_predicted_goal_value = update_metrics.mean_predicted_goal_value;
  metrics.goal_actor_loss_ratio = update_metrics.goal_actor_loss_ratio;
  metrics.mean_actual_goal_occupancy = update_metrics.mean_actual_goal_occupancy;
  metrics.goal_occupancy_correlation = update_metrics.goal_occupancy_correlation;

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

  if (update_index % config_.es_lora.es_interval == 0) {
    run_es_lora_update(update_index, metrics);
  }

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
      .architecture_name = "continuum_goal_conditioned",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
      .critic_heads = {"extrinsic"},
  };
}

void APPOTrainer::save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index) const {
  synchronize_cuda_if_needed(device_, "checkpoint save start");
  const std::filesystem::path staging = make_checkpoint_staging_directory(directory);
  remove_checkpoint_directory(staging);
  try {
    std::filesystem::create_directories(staging);
    save_experiment_config(config_, (staging / "config.json").string());
    save_checkpoint_metadata(make_checkpoint_metadata(global_step, update_index), (staging / "metadata.json").string());

    torch::NoGradGuard no_grad;
    PPOActor actor_cpu = clone_ppo_actor(actor_, torch::Device(torch::kCPU));
    ObservationNormalizer normalizer_cpu = actor_normalizer_.clone();
    normalizer_cpu.to(torch::Device(torch::kCPU));

    torch::serialize::OutputArchive actor_archive;
    actor_cpu->save(actor_archive);
    normalizer_cpu.save(actor_archive);
    actor_archive.save_to((staging / "model.pt").string());

    std::error_code ec;
    std::filesystem::remove(staging / "actor_optimizer.pt", ec);
    commit_checkpoint_directory(staging, directory);
    synchronize_cuda_if_needed(device_, "checkpoint save end");
  } catch (...) {
    remove_checkpoint_directory(staging);
    throw;
  }
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
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "dappo_train");
  std::int64_t global_step = resumed_global_step_;
  const bool train_forever = updates <= 0;
  for (int index = 0; train_forever || index < updates; ++index) {
    const int update_index = static_cast<int>(resumed_update_index_) + index + 1;
    TrainerMetrics metrics = run_update(&global_step, update_index);
    append_metrics_line(checkpoint_dir, update_index, global_step, metrics);
    std::cout << "update=" << update_index
              << " global_step=" << global_step
              << " policy_loss=" << metrics.policy_loss
              << " value_loss=" << metrics.value_loss
              << " entropy=" << metrics.entropy
              << " grad_norm=" << metrics.grad_norm
              << " epsilon=" << metrics.adaptive_epsilon
              << " sparse_reward=" << metrics.sparse_reward_mean
              << " goal_critic_loss=" << metrics.goal_critic_loss
              << " goal_actor_loss=" << metrics.goal_actor_loss
              << " goal_actor_ratio=" << metrics.goal_actor_loss_ratio
              << " mean_goal_dist=" << metrics.mean_goal_distance
              << " ball_prox=" << metrics.ball_proximity_rate
              << " goals=" << metrics.goals_scored << "/" << metrics.goals_conceded
              << " goal_corr=" << metrics.goal_occupancy_correlation
              << " es_fitness=" << metrics.es_fitness_mean
              << '\n';
    if (wandb.enabled()) {
      nlohmann::json payload{
          {"update", update_index},
          {"global_step", global_step},
          {"policy_loss", metrics.policy_loss},
          {"value_loss", metrics.value_loss},
          {"entropy", metrics.entropy},
          {"adaptive_epsilon", metrics.adaptive_epsilon},
          {"critic_variance", metrics.critic_variance},
          {"mean_confidence_weight", metrics.mean_confidence_weight},
          {"sparse_reward_mean", metrics.sparse_reward_mean},
          {"sampled_value_win_mean", metrics.sampled_value_win_mean},
          {"value_win_entropy", metrics.value_win_entropy},
          {"goal_critic_loss", metrics.goal_critic_loss},
          {"goal_actor_loss", metrics.goal_actor_loss},
          {"mean_predicted_goal_value", metrics.mean_predicted_goal_value},
          {"mean_actual_goal_occupancy", metrics.mean_actual_goal_occupancy},
          {"mean_goal_distance", metrics.mean_goal_distance},
          {"min_goal_distance", metrics.min_goal_distance},
      {"ball_proximity_rate", metrics.ball_proximity_rate},
          {"goals_scored", metrics.goals_scored},
          {"goals_conceded", metrics.goals_conceded},
          {"goal_occupancy_correlation", metrics.goal_occupancy_correlation},
          {"goal_actor_loss_ratio", metrics.goal_actor_loss_ratio},
      };
      if (update_index % config_.es_lora.es_interval == 0) {
        payload["es_fitness_mean"] = metrics.es_fitness_mean;
        payload["es_fitness_std"] = metrics.es_fitness_std;
        payload["es_fitness_best"] = metrics.es_fitness_best;
        payload["es_winrate_mean"] = metrics.es_winrate_mean;
        payload["es_goal_pressure_mean"] = metrics.es_goal_pressure_mean;
        payload["es_kl_mean"] = metrics.es_kl_mean;
        payload["es_update_norm"] = metrics.es_update_norm;
        payload["es_lora_a_norm"] = metrics.es_lora_a_norm;
        payload["es_lora_b_norm"] = metrics.es_lora_b_norm;
      }
      for (const auto& [mode, rating] : metrics.elo_ratings) {
        payload["elo_" + mode] = rating;
      }
      wandb.log(payload);
    }
    if (config_.ppo.checkpoint_interval > 0 && update_index % config_.ppo.checkpoint_interval == 0) {
      std::cout << "checkpoint_start update=" << update_index << std::endl;
      save_checkpoint(std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(update_index)), global_step, update_index);
      prune_old_checkpoints(checkpoint_dir);
      std::cout << "checkpoint_done update=" << update_index << std::endl;
    }
  }
  save_checkpoint(std::filesystem::path(checkpoint_dir) / "final", global_step, static_cast<int>(resumed_update_index_) + updates);
  wandb.finish();
}

}  // namespace pulsar

#endif
