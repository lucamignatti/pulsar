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
#include <system_error>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/training/cuda_utils.hpp"
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

torch::Tensor policy_goal_values_like(const torch::Tensor& obs, int goal_dim) {
  const auto options = obs.options().dtype(torch::kFloat32);
  if (obs.dim() == 3) {
    return torch::zeros({obs.size(0), obs.size(1), goal_dim}, options);
  }
  return torch::zeros({obs.size(0), goal_dim}, options);
}
struct OutcomeFilterStats {
  int64_t scored_episodes = 0;
  int64_t neutral_episodes = 0;
  int64_t unfinished_segments = 0;
};

struct PositionAccumulator {
  double blue_sum_x = 0, blue_sum_y = 0, blue_sum_z = 0;
  double blue_sum_x2 = 0, blue_sum_y2 = 0, blue_sum_z2 = 0;
  double orange_sum_x = 0, orange_sum_y = 0, orange_sum_z = 0;
  double orange_sum_x2 = 0, orange_sum_y2 = 0, orange_sum_z2 = 0;
  int64_t blue_count = 0, orange_count = 0;
  double blue_ball_dist_sum = 0, orange_ball_dist_sum = 0;
  double blue_intra_dist_sum = 0, orange_intra_dist_sum = 0;
  int64_t blue_intra_count = 0, orange_intra_count = 0;
  double ball_sum_x = 0, ball_sum_y = 0, ball_sum_z = 0;
  int64_t ball_count = 0;
  int64_t blue_def_third = 0, blue_mid_third = 0, blue_off_third = 0;
  int64_t orange_def_third = 0, orange_mid_third = 0, orange_off_third = 0;
  int64_t blue_ground = 0, blue_low_aerial = 0, blue_high_aerial = 0;
  int64_t orange_ground = 0, orange_low_aerial = 0, orange_high_aerial = 0;

  void accumulate(const BatchedRocketSimCollector& collector) {
    torch::Tensor car_pos = collector.host_car_positions();
    torch::Tensor ball_pos = collector.host_ball_position();
    const float* cp = car_pos.data_ptr<float>();
    const float* bp = ball_pos.data_ptr<float>();
    const int n = static_cast<int>(car_pos.size(0));

    std::vector<int> blue_idx, orange_idx;
    blue_idx.reserve(static_cast<std::size_t>(n));
    orange_idx.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
      const float x = cp[i * 4 + 0];
      const float y = cp[i * 4 + 1];
      const float z = cp[i * 4 + 2];
      const float team = cp[i * 4 + 3];
      const float bx = bp[i * 3 + 0];
      const float by = bp[i * 3 + 1];
      const float bz = bp[i * 3 + 2];

      const float dx = x - bx;
      const float dy = y - by;
      const float dz = z - bz;
      const float ball_dist = std::sqrt(dx * dx + dy * dy + dz * dz);

      if (team < 0.5F) {
        blue_sum_x += static_cast<double>(x);
        blue_sum_y += static_cast<double>(y);
        blue_sum_z += static_cast<double>(z);
        blue_sum_x2 += static_cast<double>(x) * x;
        blue_sum_y2 += static_cast<double>(y) * y;
        blue_sum_z2 += static_cast<double>(z) * z;
        blue_count++;
        blue_ball_dist_sum += static_cast<double>(ball_dist);
        blue_idx.push_back(i);
        if (y < -1707.0F) blue_def_third++;
        else if (y < 1707.0F) blue_mid_third++;
        else blue_off_third++;
        if (z < 200.0F) blue_ground++;
        else if (z < 800.0F) blue_low_aerial++;
        else blue_high_aerial++;
      } else {
        orange_sum_x += static_cast<double>(x);
        orange_sum_y += static_cast<double>(y);
        orange_sum_z += static_cast<double>(z);
        orange_sum_x2 += static_cast<double>(x) * x;
        orange_sum_y2 += static_cast<double>(y) * y;
        orange_sum_z2 += static_cast<double>(z) * z;
        orange_count++;
        orange_ball_dist_sum += static_cast<double>(ball_dist);
        orange_idx.push_back(i);
        if (y > 1707.0F) orange_def_third++;
        else if (y > -1707.0F) orange_mid_third++;
        else orange_off_third++;
        if (z < 200.0F) orange_ground++;
        else if (z < 800.0F) orange_low_aerial++;
        else orange_high_aerial++;
      }

      ball_sum_x += static_cast<double>(bx);
      ball_sum_y += static_cast<double>(by);
      ball_sum_z += static_cast<double>(bz);
      ball_count++;
    }

    for (std::size_t a = 0; a < blue_idx.size(); ++a) {
      for (std::size_t b = a + 1; b < blue_idx.size(); ++b) {
        const int ia = blue_idx[a];
        const int ib = blue_idx[b];
        const float dx = cp[ia * 4 + 0] - cp[ib * 4 + 0];
        const float dy = cp[ia * 4 + 1] - cp[ib * 4 + 1];
        const float dz = cp[ia * 4 + 2] - cp[ib * 4 + 2];
        blue_intra_dist_sum += static_cast<double>(std::sqrt(dx * dx + dy * dy + dz * dz));
        blue_intra_count++;
      }
    }
    for (std::size_t a = 0; a < orange_idx.size(); ++a) {
      for (std::size_t b = a + 1; b < orange_idx.size(); ++b) {
        const int ia = orange_idx[a];
        const int ib = orange_idx[b];
        const float dx = cp[ia * 4 + 0] - cp[ib * 4 + 0];
        const float dy = cp[ia * 4 + 1] - cp[ib * 4 + 1];
        const float dz = cp[ia * 4 + 2] - cp[ib * 4 + 2];
        orange_intra_dist_sum += static_cast<double>(std::sqrt(dx * dx + dy * dy + dz * dz));
        orange_intra_count++;
      }
    }
  }

  void finalize(TrainerMetrics& m) const {
    if (blue_count > 0) {
      m.car_pos_x_mean_blue = blue_sum_x / static_cast<double>(blue_count);
      m.car_pos_y_mean_blue = blue_sum_y / static_cast<double>(blue_count);
      m.car_pos_z_mean_blue = blue_sum_z / static_cast<double>(blue_count);
      const double var_x = blue_sum_x2 / static_cast<double>(blue_count)
          - m.car_pos_x_mean_blue * m.car_pos_x_mean_blue;
      const double var_y = blue_sum_y2 / static_cast<double>(blue_count)
          - m.car_pos_y_mean_blue * m.car_pos_y_mean_blue;
      const double var_z = blue_sum_z2 / static_cast<double>(blue_count)
          - m.car_pos_z_mean_blue * m.car_pos_z_mean_blue;
      m.car_pos_spread_blue = std::sqrt(std::max(0.0, var_x + var_y + var_z));
      m.car_ball_distance_mean_blue = blue_ball_dist_sum / static_cast<double>(blue_count);
      m.blue_defensive_third_rate = static_cast<double>(blue_def_third) / static_cast<double>(blue_count);
      m.blue_midfield_third_rate = static_cast<double>(blue_mid_third) / static_cast<double>(blue_count);
      m.blue_offensive_third_rate = static_cast<double>(blue_off_third) / static_cast<double>(blue_count);
      m.blue_ground_rate = static_cast<double>(blue_ground) / static_cast<double>(blue_count);
      m.blue_low_aerial_rate = static_cast<double>(blue_low_aerial) / static_cast<double>(blue_count);
      m.blue_high_aerial_rate = static_cast<double>(blue_high_aerial) / static_cast<double>(blue_count);
    }
    if (orange_count > 0) {
      m.car_pos_x_mean_orange = orange_sum_x / static_cast<double>(orange_count);
      m.car_pos_y_mean_orange = orange_sum_y / static_cast<double>(orange_count);
      m.car_pos_z_mean_orange = orange_sum_z / static_cast<double>(orange_count);
      const double var_x = orange_sum_x2 / static_cast<double>(orange_count)
          - m.car_pos_x_mean_orange * m.car_pos_x_mean_orange;
      const double var_y = orange_sum_y2 / static_cast<double>(orange_count)
          - m.car_pos_y_mean_orange * m.car_pos_y_mean_orange;
      const double var_z = orange_sum_z2 / static_cast<double>(orange_count)
          - m.car_pos_z_mean_orange * m.car_pos_z_mean_orange;
      m.car_pos_spread_orange = std::sqrt(std::max(0.0, var_x + var_y + var_z));
      m.car_ball_distance_mean_orange = orange_ball_dist_sum / static_cast<double>(orange_count);
      m.orange_defensive_third_rate = static_cast<double>(orange_def_third) / static_cast<double>(orange_count);
      m.orange_midfield_third_rate = static_cast<double>(orange_mid_third) / static_cast<double>(orange_count);
      m.orange_offensive_third_rate = static_cast<double>(orange_off_third) / static_cast<double>(orange_count);
      m.orange_ground_rate = static_cast<double>(orange_ground) / static_cast<double>(orange_count);
      m.orange_low_aerial_rate = static_cast<double>(orange_low_aerial) / static_cast<double>(orange_count);
      m.orange_high_aerial_rate = static_cast<double>(orange_high_aerial) / static_cast<double>(orange_count);
    }
    if (blue_intra_count > 0) {
      m.car_intra_team_distance_blue = blue_intra_dist_sum / static_cast<double>(blue_intra_count);
    }
    if (orange_intra_count > 0) {
      m.car_intra_team_distance_orange = orange_intra_dist_sum / static_cast<double>(orange_intra_count);
    }
    if (ball_count > 0) {
      m.ball_pos_x_mean = ball_sum_x / static_cast<double>(ball_count);
      m.ball_pos_y_mean = ball_sum_y / static_cast<double>(ball_count);
      m.ball_pos_z_mean = ball_sum_z / static_cast<double>(ball_count);
    }
  }
};

void zero_active_segment(
    float* learner_active,
    int total_agents,
    int start_step,
    int end_step,
    int agent_begin,
    int agent_count) {
  if (end_step < start_step || agent_count <= 0) {
    return;
  }
  for (int step = start_step; step <= end_step; ++step) {
    float* row = learner_active + static_cast<std::ptrdiff_t>(step * total_agents + agent_begin);
    std::fill(row, row + agent_count, 0.0F);
  }
}

OutcomeFilterStats keep_only_scored_episode_segments(RolloutStorage& rollout, int agents_per_env) {
  OutcomeFilterStats stats{};
  const int steps = rollout.rollout_length();
  if (steps <= 0) {
    return stats;
  }
  const int total_agents = rollout.num_agents();
  float* learner_active = rollout.learner_active.data_ptr<float>();
  const float* starts = rollout.episode_starts.data_ptr<float>();
  const float* dones = rollout.dones.data_ptr<float>();
  const std::int64_t* labels = rollout.terminal_outcome_labels.data_ptr<std::int64_t>();
  for (int agent_begin = 0; agent_begin < total_agents; agent_begin += agents_per_env) {
    const int agent_count = std::min(agents_per_env, total_agents - agent_begin);
    int segment_start = 0;
    for (int step = 0; step < steps; ++step) {
      const int row_offset = step * total_agents + agent_begin;
      if (step > segment_start && starts[row_offset] > 0.5F) {
        zero_active_segment(learner_active, total_agents, segment_start, step - 1, agent_begin, agent_count);
        stats.unfinished_segments++;
        segment_start = step;
      }
      if (dones[row_offset] <= 0.5F) {
        continue;
      }
      bool scored = false;
      for (int local = 0; local < agent_count; ++local) {
        scored = scored || (labels[row_offset + local] == 0 || labels[row_offset + local] == 1);
      }
      if (scored) {
        stats.scored_episodes++;
      } else {
        zero_active_segment(learner_active, total_agents, segment_start, step, agent_begin, agent_count);
        stats.neutral_episodes++;
      }
      segment_start = step + 1;
    }
    if (segment_start < steps) {
      zero_active_segment(learner_active, total_agents, segment_start, steps - 1, agent_begin, agent_count);
      stats.unfinished_segments++;
    }
  }
  return stats;
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
      {"sparse_reward_mean", metrics.sparse_reward_mean},
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
      {"car_pos_x_mean_blue", metrics.car_pos_x_mean_blue},
      {"car_pos_y_mean_blue", metrics.car_pos_y_mean_blue},
      {"car_pos_z_mean_blue", metrics.car_pos_z_mean_blue},
      {"car_pos_x_mean_orange", metrics.car_pos_x_mean_orange},
      {"car_pos_y_mean_orange", metrics.car_pos_y_mean_orange},
      {"car_pos_z_mean_orange", metrics.car_pos_z_mean_orange},
      {"car_pos_spread_blue", metrics.car_pos_spread_blue},
      {"car_pos_spread_orange", metrics.car_pos_spread_orange},
      {"car_ball_distance_mean_blue", metrics.car_ball_distance_mean_blue},
      {"car_ball_distance_mean_orange", metrics.car_ball_distance_mean_orange},
      {"car_intra_team_distance_blue", metrics.car_intra_team_distance_blue},
      {"car_intra_team_distance_orange", metrics.car_intra_team_distance_orange},
      {"ball_pos_x_mean", metrics.ball_pos_x_mean},
      {"ball_pos_y_mean", metrics.ball_pos_y_mean},
      {"ball_pos_z_mean", metrics.ball_pos_z_mean},
      {"blue_defensive_third_rate", metrics.blue_defensive_third_rate},
      {"blue_midfield_third_rate", metrics.blue_midfield_third_rate},
      {"blue_offensive_third_rate", metrics.blue_offensive_third_rate},
      {"orange_defensive_third_rate", metrics.orange_defensive_third_rate},
      {"orange_midfield_third_rate", metrics.orange_midfield_third_rate},
      {"orange_offensive_third_rate", metrics.orange_offensive_third_rate},
      {"blue_ground_rate", metrics.blue_ground_rate},
      {"blue_low_aerial_rate", metrics.blue_low_aerial_rate},
      {"blue_high_aerial_rate", metrics.blue_high_aerial_rate},
      {"orange_ground_rate", metrics.orange_ground_rate},
      {"orange_low_aerial_rate", metrics.orange_low_aerial_rate},
      {"orange_high_aerial_rate", metrics.orange_high_aerial_rate},
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

ContinuumState concatenate_states(const std::vector<ContinuumState>& states) {
  if (states.empty()) {
    return {};
  }
  if (states.size() == 1) {
    return clone_state(states.front());
  }
  std::vector<torch::Tensor> workspaces;
  std::vector<torch::Tensor> stm_keys;
  std::vector<torch::Tensor> stm_values;
  std::vector<torch::Tensor> stm_strengths;
  std::vector<torch::Tensor> stm_write_indices;
  std::vector<torch::Tensor> ltm_coeffs;
  std::vector<torch::Tensor> timesteps;
  workspaces.reserve(states.size());
  stm_keys.reserve(states.size());
  stm_values.reserve(states.size());
  stm_strengths.reserve(states.size());
  stm_write_indices.reserve(states.size());
  ltm_coeffs.reserve(states.size());
  timesteps.reserve(states.size());
  for (const auto& state : states) {
    workspaces.push_back(state.workspace);
    stm_keys.push_back(state.stm_keys);
    stm_values.push_back(state.stm_values);
    stm_strengths.push_back(state.stm_strengths);
    stm_write_indices.push_back(state.stm_write_index);
    ltm_coeffs.push_back(state.ltm_coeffs);
    timesteps.push_back(state.timestep);
  }
  return {
      torch::cat(workspaces, 0),
      torch::cat(stm_keys, 0),
      torch::cat(stm_values, 0),
      torch::cat(stm_strengths, 0),
      torch::cat(stm_write_indices, 0),
      torch::cat(ltm_coeffs, 0),
      torch::cat(timesteps, 0),
  };
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
      action_table_(config_.action_table),
      actor_(PPOActor(config_.model, config_.goal_critic)),
      actor_normalizer_(config_.model.observation_dim),
      actor_optimizer_(actor_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate).eps(1.0e-5F)),
      rollout_(make_rollout_storage(
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
  total_agents_ = total_agents_for_collectors(collectors_);
  if (total_agents_ == 0) {
    throw std::invalid_argument("APPOTrainer collectors must contain agents.");
  }
  seed_everything(config_.env.seed);
  collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  opponent_collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  configure_cuda_runtime(device_);
  use_pinned_host_buffers_ = device_.is_cuda();
  actor_->to(device_);
  actor_normalizer_.to(device_);

  maybe_initialize_from_checkpoint();

  shard_agent_offsets_.clear();
  shard_collection_states_.clear();
  shard_opponent_collection_states_.clear();
  std::int64_t agent_offset = 0;
  for (const auto& collector : collectors_) {
    if (!collector) {
      throw std::invalid_argument("APPOTrainer collectors must be non-null.");
    }
    shard_agent_offsets_.push_back(agent_offset);
    const auto shard_agents = static_cast<std::int64_t>(collector->total_agents());
    shard_collection_states_.push_back(actor_->initial_state(shard_agents, device_));
    shard_opponent_collection_states_.push_back(actor_->initial_state(shard_agents, device_));
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

torch::Tensor APPOTrainer::map_outcome_labels_to_rewards(const torch::Tensor& labels) const {
  torch::Tensor rewards = torch::zeros_like(labels, torch::TensorOptions().dtype(torch::kFloat32));
  rewards.masked_fill_(labels == 0, config_.outcome.score);
  rewards.masked_fill_(labels == 1, config_.outcome.concede);
  rewards.masked_fill_(labels == 2, config_.outcome.neutral);
  rewards.masked_fill_(labels == 3, config_.outcome.neutral_no_touch);
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
  PULSAR_TRACE_SCOPE_CAT("trainer", "update_actor");
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  const int seq_len = std::max(1, config_.ppo.sequence_length);
  const int agents_per_batch = std::max(1, config_.ppo.minibatch_size / seq_len);
  const int total_agents = rollout_.num_agents();
  const int rollout_steps = rollout_.rollout_length();
  std::int64_t metric_steps = 0;
  double accumulated_goal_critic_loss = 0.0;
  double accumulated_goal_score = 0.0;
  double accumulated_sampled_goal_distance = 0.0;

  const auto& all_values = rollout_.all_values();
  const auto& all_rewards = rollout_.all_rewards();
  if (rollout_steps <= 0) {
    return metrics;
  }
  const torch::Tensor extrinsic_values = all_values.at("extrinsic").narrow(0, 0, rollout_steps);
  const torch::Tensor extrinsic_rewards = all_rewards.at("extrinsic").narrow(0, 0, rollout_steps);
  const torch::Tensor rollout_dones = rollout_.dones.narrow(0, 0, rollout_steps);

  torch::Tensor active_mask = rollout_.learner_active.narrow(0, 0, rollout_steps) > 0.5F;
  torch::Tensor sparse_advantages;
  torch::Tensor normalized_advantages;
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "update_gae");
    sparse_advantages = compute_gae(
      extrinsic_values,
      extrinsic_rewards,
      rollout_dones,
      config_.ppo.gamma,
      config_.ppo.gae_lambda,
      rollout_.final_values().count("extrinsic") ? rollout_.final_values().at("extrinsic") : torch::Tensor{});
    normalized_advantages = normalize_advantage(sparse_advantages, active_mask);
  }
  torch::Tensor sparse_returns = sparse_advantages + extrinsic_values.detach();

  for (int epoch = 0; epoch < config_.ppo.update_epochs; ++epoch) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "update_epoch");
    const torch::Tensor perm = torch::randperm(total_agents, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    for (int agent_offset = 0; agent_offset < total_agents; agent_offset += agents_per_batch) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "update_minibatch");
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
        ActorSequenceOutput output;
        {
          PULSAR_TRACE_SCOPE_CAT("trainer", "update_forward_sequence");
          const torch::Tensor goal_values = policy_goal_values_like(obs, config_.goal_critic.goal_dim);
          output = actor_->forward_sequence(obs, std::move(state), episode_starts, goal_values);
        }
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

        torch::Tensor policy_loss =
            clipped_ppo_policy_loss(current_log_probs, active_old_log_probs, active_advantages, epsilon);
        policy_loss = policy_loss.mean();

        const torch::Tensor entropy = masked_action_entropy(active_logits, active_masks).mean();
        torch::Tensor entropy_floor_loss = torch::zeros({}, active_advantages.options());
        if (config_.ppo.entropy_floor > 0.0F && config_.ppo.entropy_floor_coef > 0.0F) {
          const torch::Tensor entropy_floor = torch::full_like(entropy, config_.ppo.entropy_floor);
          entropy_floor_loss = config_.ppo.entropy_floor_coef * torch::relu(entropy_floor - entropy).square();
        }

        torch::Tensor chunk_returns =
            sparse_returns.narrow(0, loss_start, loss_steps).index_select(1, agent_indices).to(device_).reshape({samples});
        torch::Tensor active_returns = chunk_returns.index({flat_active});

        torch::Tensor value_win_chunk = output.value_win_logits.narrow(0, burn, loss_steps);
        torch::Tensor flat_value_win_logits = value_win_chunk.reshape({samples, 1});
        torch::Tensor active_value_win_logits = flat_value_win_logits.index({flat_active});

        torch::Tensor value_loss = torch::mse_loss(
            active_value_win_logits.squeeze(-1), active_returns, torch::Reduction::Mean);

        torch::Tensor goal_loss = torch::zeros({}, active_advantages.options());
        torch::Tensor actor_goal_loss = torch::zeros({}, active_advantages.options());
        double chunk_goal_score = 0.0;

        {
          torch::Tensor chunk_goal_pos =
              rollout_.goal_positions.narrow(0, loss_start, loss_steps).index_select(1, agent_indices);
          torch::Tensor chunk_dones =
              rollout_.dones.narrow(0, loss_start, loss_steps).index_select(1, agent_indices);
          torch::Tensor chunk_ep_starts =
              rollout_.episode_starts.narrow(0, loss_start, loss_steps).index_select(1, agent_indices);
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
                feat_sub.detach(), sampled.detach(), goal_sub).mean();
          } else {
            sa_emb = actor_->goal_critic()->sa_embedding(active_features, active_actions);
            g_emb = actor_->goal_critic()->goal_embedding(active_future_goal_pos);

            torch::Tensor sampled = sample_masked_actions(
                active_logits.detach(), active_masks.detach(), false, nullptr);
            actor_goal_loss = -actor_->goal_critic()->forward(
                active_features.detach(), sampled.detach(), active_future_goal_pos).mean();
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

        const torch::Tensor loss =
            policy_loss
            + config_.ppo.value_coef * value_loss
            + config_.goal_critic.lambda_Zg * goal_loss
            + config_.goal_critic.lambda_goal_actor * actor_goal_loss
            + entropy_floor_loss
            - config_.ppo.entropy_coef * entropy;

        metrics.forward_backward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - forward_start).count();

        require_finite(loss, "loss");
        require_finite(policy_loss, "policy_loss");
        require_finite(value_loss, "value_loss");
        require_finite(entropy, "entropy");
        require_finite(entropy_floor_loss, "entropy_floor_loss");
        require_finite(goal_loss, "goal_loss");

        const auto active_samples = active_logits.size(0);
        const torch::Tensor weighted_loss = loss * (static_cast<double>(active_samples) / total_active_samples_agent);
        {
          PULSAR_TRACE_SCOPE_CAT("trainer", "update_backward");
          weighted_loss.backward();
        }

        metrics.policy_loss += policy_loss.item<double>() * static_cast<double>(active_samples);
        metrics.value_loss += value_loss.item<double>() * static_cast<double>(active_samples);
        metrics.entropy += entropy.item<double>() * static_cast<double>(active_samples);
        metric_steps += active_samples;
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

      const torch::Tensor goal_values = policy_goal_values_like(normalized_obs, config_.goal_critic.goal_dim);
      ActorStepOutput output = actor_->forward_step(normalized_obs, std::move(eval_state), episode_starts, goal_values);
      eval_state = std::move(output.state);
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

  const float step = es_cfg.eta_ES * es_cfg.sigma_ES;
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

  float best_fitness = *std::max_element(fitnesses.begin(), fitnesses.end());

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

TrainerMetrics APPOTrainer::run_update(std::int64_t* global_step, int update_index) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "run_update");
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  CollectorTimings collector_timings{};
  BatchedRocketSimCollector* collector_ = collectors_.front().get();
  std::int64_t collected_agent_steps = 0;
  PositionAccumulator pos_acc{};

  const auto collection_start = std::chrono::steady_clock::now();
  if (config_.ppo.train_only_scored_episodes) {
    if (collectors_.size() > 1) {
      for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
        collectors_[shard]->reset_all(&collector_timings);
        const auto shard_agents = static_cast<std::int64_t>(collectors_[shard]->total_agents());
        shard_collection_states_[shard] = actor_->initial_state(shard_agents, device_);
        shard_opponent_collection_states_[shard] = actor_->initial_state(shard_agents, device_);
      }
    } else {
      collector_->reset_all(&collector_timings);
      collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
      opponent_collection_state_ = actor_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
    }
  }
  rollout_.set_initial_state(
      collectors_.size() > 1 ? concatenate_states(shard_collection_states_) : collection_state_);

  double total_sparse_reward = 0.0;
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
  const int agents_per_env = std::max(1, config_.env.team_size * 2);
  const int min_rollout_steps = config_.ppo.min_rollout_length > 0
      ? config_.ppo.min_rollout_length
      : config_.ppo.rollout_length;
  const int early_update_completed_episodes = config_.ppo.early_update_completed_episodes;

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

    const bool use_completed_for_break =
        !config_.ppo.train_only_scored_episodes;

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
          output = actor_->forward_step(normalized_obs, std::move(shard_collection_states_[shard]), episode_starts, goal_values);
          shard_collection_states_[shard] = std::move(output.state);
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
              actor_,
              raw_obs,
              action_masks,
              episode_starts,
              snapshot_ids,
              shard_opponent_collection_states_[shard],
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
        torch::Tensor terminal_labels = collector.host_terminal_outcome_labels();
        torch::Tensor extrinsic_rewards_host = map_outcome_labels_to_rewards(terminal_labels) * dones_host;
        const auto* dones_ptr = dones_host.data_ptr<float>();

        torch::Tensor ball_prox_host = collector.host_ball_proximity();
        total_ball_proximity_steps += ball_prox_host.sum().item<int64_t>();
        total_ball_proximity_denom += ball_prox_host.numel();

        const auto* tl_ptr = terminal_labels.data_ptr<std::int64_t>();
        const auto* la_ptr = shard_step.learner_active_host.data_ptr<float>();
        for (int64_t i = 0; i < terminal_labels.numel(); ++i) {
          if (la_ptr[i] > 0.5F && dones_ptr[i] > 0.5F) {
            if (tl_ptr[i] == 0) {
              total_goals_scored++;
            } else if (tl_ptr[i] == 1) {
              total_goals_conceded++;
            }
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

        const auto learner_step_count = static_cast<std::int64_t>(shard_step.learner_active_host.sum().item<float>());
        total_sparse_reward += extrinsic_rewards_host.sum().item<double>();
        total_steps += extrinsic_rewards_host.numel();
        total_learner_steps += learner_step_count;

        std::unordered_map<std::string, torch::Tensor> all_values;
        all_values["extrinsic"] = sampled_value_cpu;

        std::unordered_map<std::string, torch::Tensor> all_rewards;
        all_rewards["extrinsic"] = extrinsic_rewards_host;

        rollout_.append_slice(
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
            goal_pos_host,
            terminal_labels);

        pos_acc.accumulate(collector);
        if (heatmap_logger_.enabled()) {
          torch::Tensor cp = collector.host_car_positions();
          heatmap_logger_.record_positions_interleaved(
              cp.data_ptr<float>(), static_cast<int>(cp.size(0)));
        }

        collected_agent_steps += learner_step_count;
      }

      if (early_update_completed_episodes > 0
          && step + 1 >= min_rollout_steps
          && (use_completed_for_break ? completed_episodes : scored_episodes) >= early_update_completed_episodes) {
        break;
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
        ContinuumState bootstrap_state = clone_state(shard_collection_states_[shard]);
        const torch::Tensor final_goal_values = policy_goal_values_like(final_normalized, config_.goal_critic.goal_dim);
        ActorStepOutput final_output = actor_->forward_step(
            final_normalized, std::move(bootstrap_state), final_starts, final_goal_values);
        final_values.push_back(final_output.value_win_logits.squeeze(-1).to(torch::kCPU));
      }
      std::unordered_map<std::string, torch::Tensor> bootstrap_values;
      bootstrap_values["extrinsic"] = torch::cat(final_values, 0);
      rollout_.set_final_values(bootstrap_values);
    }
  } else {
    PULSAR_TRACE_SCOPE_CAT("trainer", "collect_loop");
    const bool use_completed_for_break =
        !config_.ppo.train_only_scored_episodes;
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
      output = actor_->forward_step(normalized_obs, std::move(collection_state_), episode_starts, goal_values);
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
      torch::Tensor snapshot_ids = collector_->host_snapshot_ids().to(device_, use_pinned_host_buffers_);
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
    torch::Tensor terminal_labels = collector_->host_terminal_outcome_labels();
    torch::Tensor extrinsic_rewards_host = map_outcome_labels_to_rewards(terminal_labels) * dones_host;
    const auto* dones_ptr = dones_host.data_ptr<float>();

    torch::Tensor ball_prox_host = collector_->host_ball_proximity();
    total_ball_proximity_steps += ball_prox_host.sum().item<int64_t>();
    total_ball_proximity_denom += ball_prox_host.numel();

    const auto* tl_ptr = terminal_labels.data_ptr<std::int64_t>();
    const auto* la_ptr = learner_active_host.data_ptr<float>();
    for (int64_t i = 0; i < terminal_labels.numel(); ++i) {
      if (la_ptr[i] > 0.5F && dones_ptr[i] > 0.5F) {
        if (tl_ptr[i] == 0) {
          total_goals_scored++;
        } else if (tl_ptr[i] == 1) {
          total_goals_conceded++;
        }
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

    const auto learner_step_count = static_cast<std::int64_t>(learner_active_host.sum().item<float>());
    total_sparse_reward += extrinsic_rewards_host.sum().item<double>();
    total_steps += extrinsic_rewards_host.numel();
    total_learner_steps += learner_step_count;

    std::unordered_map<std::string, torch::Tensor> all_values;
    all_values["extrinsic"] = sampled_value_cpu;

    std::unordered_map<std::string, torch::Tensor> all_rewards;
    all_rewards["extrinsic"] = extrinsic_rewards_host;

    rollout_.append(
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
        goal_pos_host,
        terminal_labels);

    pos_acc.accumulate(*collector_);
    if (heatmap_logger_.enabled()) {
      torch::Tensor cp = collector_->host_car_positions();
      heatmap_logger_.record_positions_interleaved(
          cp.data_ptr<float>(), static_cast<int>(cp.size(0)));
    }

    collected_agent_steps += learner_step_count;
    if (early_update_completed_episodes > 0
        && step + 1 >= min_rollout_steps
        && (use_completed_for_break ? completed_episodes : scored_episodes) >= early_update_completed_episodes) {
      break;
    }
    }
  }
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "bootstrap_forward");
    torch::NoGradGuard no_grad;
    torch::Tensor final_raw_obs = collector_->host_observations().to(device_, use_pinned_host_buffers_);
    torch::Tensor final_normalized = actor_normalizer_.normalize(final_raw_obs);
    torch::Tensor final_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    ContinuumState bootstrap_state = clone_state(collection_state_);
    const torch::Tensor final_goal_values = policy_goal_values_like(final_normalized, config_.goal_critic.goal_dim);
    ActorStepOutput final_output = actor_->forward_step(
        final_normalized, std::move(bootstrap_state), final_starts, final_goal_values);

    std::unordered_map<std::string, torch::Tensor> bootstrap_values;
    bootstrap_values["extrinsic"] = final_output.value_win_logits.squeeze(-1).to(torch::kCPU);
    rollout_.set_final_values(bootstrap_values);
    }
  }

  OutcomeFilterStats outcome_filter_stats{};
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "scored_filter");
    if (config_.ppo.train_only_scored_episodes) {
      outcome_filter_stats = keep_only_scored_episode_segments(rollout_, agents_per_env);
      const torch::Tensor active_train_mask = rollout_.learner_active.narrow(0, 0, rollout_.rollout_length());
      const torch::Tensor reward_train = rollout_.reward("extrinsic").narrow(0, 0, rollout_.rollout_length());
      total_learner_steps = active_train_mask.sum().item<int64_t>();
      total_sparse_reward = (reward_train * active_train_mask).sum().item<double>();
    }
  }

  const double collection_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

  if (total_learner_steps > 0) {
    metrics.sparse_reward_mean = total_sparse_reward / static_cast<double>(total_learner_steps);
    metrics.mean_goal_distance = total_goal_distance / static_cast<double>(std::max(rollout_.rollout_length(), 1));
  }
  metrics.min_goal_distance = min_goal_distance;
  metrics.goals_scored = total_goals_scored;
  metrics.goals_conceded = total_goals_conceded;
  metrics.rollout_steps = rollout_.rollout_length();
  metrics.completed_episodes = completed_episodes;
  const bool filtered = config_.ppo.train_only_scored_episodes;
  metrics.scored_episodes = filtered ? outcome_filter_stats.scored_episodes : scored_episodes;
  if (total_ball_proximity_denom > 0) {
    metrics.ball_proximity_rate = static_cast<double>(total_ball_proximity_steps) / static_cast<double>(total_ball_proximity_denom);
  }
  if (accumulated_value_count > 0) {
    metrics.sampled_value_win_mean = accumulated_sampled_value
        / static_cast<double>(accumulated_value_count);
  }

  TrainerMetrics update_metrics = update_actor();
  metrics.policy_loss = update_metrics.policy_loss;
  metrics.value_loss = update_metrics.value_loss;
  metrics.entropy = update_metrics.entropy;
  metrics.grad_norm = update_metrics.grad_norm;
  metrics.update_seconds = update_metrics.update_seconds;
  metrics.forward_backward_seconds = update_metrics.forward_backward_seconds;
  metrics.optimizer_step_seconds = update_metrics.optimizer_step_seconds;
  metrics.goal_critic_loss = update_metrics.goal_critic_loss;
  metrics.mean_goal_score = update_metrics.mean_goal_score;
  metrics.mean_sampled_goal_distance = update_metrics.mean_sampled_goal_distance;

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
  pos_acc.finalize(metrics);
  heatmap_logger_.on_update_complete();
  return metrics;
}

CheckpointMetadata APPOTrainer::make_checkpoint_metadata(std::int64_t global_step, int update_index) const {
  return CheckpointMetadata{
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_hash(config_.action_table),
      .architecture_name = "continuum_contrastive_goal_appo",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
      .critic_heads = {"extrinsic"},
  };
}

void APPOTrainer::save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index) const {
  PULSAR_TRACE_SCOPE_CAT("trainer", "checkpoint_save");
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
  if (wandb.enabled()) {
    heatmap_logger_.start();
  }
  std::int64_t global_step = resumed_global_step_;
  const bool train_forever = updates <= 0;
  for (int index = 0; train_forever || index < updates; ++index) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "train_iteration");
    const int update_index = static_cast<int>(resumed_update_index_) + index + 1;
    TrainerMetrics metrics = run_update(&global_step, update_index);
    append_metrics_line(checkpoint_dir, update_index, global_step, metrics);
    std::cout << "update=" << update_index
              << " global_step=" << global_step
              << " policy_loss=" << metrics.policy_loss
              << " value_loss=" << metrics.value_loss
              << " entropy=" << metrics.entropy
              << " grad_norm=" << metrics.grad_norm
              << " sparse_reward=" << metrics.sparse_reward_mean
              << " rollout_steps=" << metrics.rollout_steps
              << " completed_eps=" << metrics.completed_episodes
              << " scored_eps=" << metrics.scored_episodes
              << " sampled_goal_dist=" << metrics.mean_sampled_goal_distance
              << " mean_goal_dist=" << metrics.mean_goal_distance
              << " ball_prox=" << metrics.ball_proximity_rate
              << " goals=" << metrics.goals_scored << "/" << metrics.goals_conceded
              << " es_fitness=" << metrics.es_fitness_mean
              << '\n';
    if (wandb.enabled()) {
      nlohmann::json payload{
          {"update", update_index},
          {"global_step", global_step},
          {"policy_loss", metrics.policy_loss},
          {"value_loss", metrics.value_loss},
          {"entropy", metrics.entropy},
          {"sparse_reward_mean", metrics.sparse_reward_mean},
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
          {"car_pos_x_mean_blue", metrics.car_pos_x_mean_blue},
          {"car_pos_y_mean_blue", metrics.car_pos_y_mean_blue},
          {"car_pos_z_mean_blue", metrics.car_pos_z_mean_blue},
          {"car_pos_x_mean_orange", metrics.car_pos_x_mean_orange},
          {"car_pos_y_mean_orange", metrics.car_pos_y_mean_orange},
          {"car_pos_z_mean_orange", metrics.car_pos_z_mean_orange},
          {"car_pos_spread_blue", metrics.car_pos_spread_blue},
          {"car_pos_spread_orange", metrics.car_pos_spread_orange},
          {"car_ball_distance_mean_blue", metrics.car_ball_distance_mean_blue},
          {"car_ball_distance_mean_orange", metrics.car_ball_distance_mean_orange},
          {"car_intra_team_distance_blue", metrics.car_intra_team_distance_blue},
          {"car_intra_team_distance_orange", metrics.car_intra_team_distance_orange},
          {"ball_pos_x_mean", metrics.ball_pos_x_mean},
          {"ball_pos_y_mean", metrics.ball_pos_y_mean},
          {"ball_pos_z_mean", metrics.ball_pos_z_mean},
          {"blue_defensive_third_rate", metrics.blue_defensive_third_rate},
          {"blue_midfield_third_rate", metrics.blue_midfield_third_rate},
          {"blue_offensive_third_rate", metrics.blue_offensive_third_rate},
          {"orange_defensive_third_rate", metrics.orange_defensive_third_rate},
          {"orange_midfield_third_rate", metrics.orange_midfield_third_rate},
          {"orange_offensive_third_rate", metrics.orange_offensive_third_rate},
          {"blue_ground_rate", metrics.blue_ground_rate},
          {"blue_low_aerial_rate", metrics.blue_low_aerial_rate},
          {"blue_high_aerial_rate", metrics.blue_high_aerial_rate},
          {"orange_ground_rate", metrics.orange_ground_rate},
          {"orange_low_aerial_rate", metrics.orange_low_aerial_rate},
          {"orange_high_aerial_rate", metrics.orange_high_aerial_rate},
      };
      if (update_index % config_.es_lora.es_interval == 0) {
        payload["es_fitness_mean"] = metrics.es_fitness_mean;
        payload["es_fitness_std"] = metrics.es_fitness_std;
        payload["es_fitness_best"] = metrics.es_fitness_best;
        payload["es_winrate_mean"] = metrics.es_winrate_mean;
        payload["es_kl_mean"] = metrics.es_kl_mean;
        payload["es_update_norm"] = metrics.es_update_norm;
        payload["es_lora_a_norm"] = metrics.es_lora_a_norm;
        payload["es_lora_b_norm"] = metrics.es_lora_b_norm;
      }
      for (const auto& [mode, rating] : metrics.elo_ratings) {
        payload["elo_" + mode] = rating;
      }
      wandb.log(payload);
      nlohmann::json heatmap_payload;
      if (heatmap_logger_.try_get_payload(heatmap_payload)) {
        wandb.log(heatmap_payload);
      }
    }
    if (config_.ppo.checkpoint_interval > 0 && update_index % config_.ppo.checkpoint_interval == 0) {
      std::cout << "checkpoint_start update=" << update_index << std::endl;
      save_checkpoint(std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(update_index)), global_step, update_index);
      prune_old_checkpoints(checkpoint_dir);
      std::cout << "checkpoint_done update=" << update_index << std::endl;
    }
  }
  save_checkpoint(std::filesystem::path(checkpoint_dir) / "final", global_step, static_cast<int>(resumed_update_index_) + updates);
  heatmap_logger_.stop();
  wandb.finish();
}

}  // namespace pulsar

#endif
