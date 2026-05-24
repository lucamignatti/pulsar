#include "pulsar/training/batched_rocketsim_collector.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#include "pulsar/rl/action_table.hpp"
#include "pulsar/tracing/tracing.hpp"

namespace pulsar {
namespace {

std::shared_ptr<MutatorSequence> make_default_reset_mutator(const EnvConfig& config) {
  return std::make_shared<MutatorSequence>(
      std::vector<StateMutatorPtr>{
          std::make_shared<FixedTeamSizeMutator>(config),
          std::make_shared<KickoffMutator>(config),
      });
}

std::vector<TransitionEnginePtr> make_default_engines(const ExperimentConfig& config) {
  std::vector<TransitionEnginePtr> engines;
  const auto reset_mutator = make_default_reset_mutator(config.env);
  engines.reserve(static_cast<std::size_t>(config.ppo.num_envs));
  for (int env_idx = 0; env_idx < config.ppo.num_envs; ++env_idx) {
    EnvConfig env_config = config.env;
    env_config.seed += static_cast<std::uint64_t>(env_idx);
    engines.push_back(std::make_shared<RocketSimTransitionEngine>(env_config, reset_mutator));
  }
  return engines;
}

void compute_goal_position(const EnvState& state, const GoalMappingConfig& cfg, float* out) {
  const BallState& ball = state.ball;
  out[0] = ball.position.x / cfg.arena_max_distance;
  out[1] = ball.position.y / cfg.arena_max_distance;
  out[2] = ball.position.z / cfg.arena_max_distance;
}

std::uint64_t mix_obs_order_seed(std::uint64_t seed, std::size_t env_idx) {
  std::uint64_t x = seed + 0x9e3779b97f4a7c15ULL + (static_cast<std::uint64_t>(env_idx) << 6U);
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31U);
}

}  // namespace

BatchedRocketSimCollector::BatchedRocketSimCollector(
    ExperimentConfig config,
    ObsBuilderPtr obs_builder,
    ActionParserPtr action_parser,
    DoneConditionPtr done_condition,
    bool pin_host_memory)
    : config_(std::move(config)),
      obs_builder_(std::move(obs_builder)),
      action_parser_(std::move(action_parser)),
      done_condition_(std::move(done_condition)),
      reward_engine_(config_),
      executor_(static_cast<std::size_t>(config_.ppo.collection_workers)) {
  initialize(make_default_engines(config_), pin_host_memory);
}

BatchedRocketSimCollector::BatchedRocketSimCollector(
    ExperimentConfig config,
    std::vector<TransitionEnginePtr> engines,
    ObsBuilderPtr obs_builder,
    ActionParserPtr action_parser,
    DoneConditionPtr done_condition,
    bool pin_host_memory)
    : config_(std::move(config)),
      obs_builder_(std::move(obs_builder)),
      action_parser_(std::move(action_parser)),
      done_condition_(std::move(done_condition)),
      reward_engine_(config_),
      executor_(static_cast<std::size_t>(config_.ppo.collection_workers)) {
  initialize(std::move(engines), pin_host_memory);
}

void BatchedRocketSimCollector::initialize(
    std::vector<TransitionEnginePtr> engines,
    bool pin_host_memory) {
  PULSAR_TRACE_SCOPE_CAT("collector", "initialize");
  if (!obs_builder_ || !action_parser_ || !done_condition_) {
    throw std::invalid_argument("BatchedRocketSimCollector requires non-null components.");
  }

  if (engines.empty()) {
    throw std::invalid_argument("BatchedRocketSimCollector requires at least one engine.");
  }

  envs_.reserve(engines.size());
  for (std::size_t env_idx = 0; env_idx < engines.size(); ++env_idx) {
    if (!engines[env_idx]) {
      throw std::invalid_argument("BatchedRocketSimCollector requires non-null engines.");
    }

    const std::size_t agent_count = engines[env_idx]->num_agents();
    envs_.push_back(EnvRuntime{
        .engine = std::move(engines[env_idx]),
        .assignment = {},
        .reset_seed = config_.env.seed + static_cast<std::uint64_t>(env_idx),
        .obs_car_order = {},
        .action_scratch = std::vector<ControllerState>(agent_count),
        .terminated_scratch = std::vector<std::uint8_t>(agent_count, 0),
        .truncated_scratch = std::vector<std::uint8_t>(agent_count, 0),
    });
  }

  agent_offsets_.reserve(envs_.size() + 1);
  agent_offsets_.push_back(0);
  for (const auto& env : envs_) {
    agent_offsets_.push_back(agent_offsets_.back() + env.engine->num_agents());
  }
  total_agents_ = agent_offsets_.back();
  obs_dim_ = static_cast<int>(obs_builder_->obs_dim());

  auto* discrete = dynamic_cast<const DiscreteActionParser*>(action_parser_.get());
  if (discrete == nullptr) {
    throw std::invalid_argument("BatchedRocketSimCollector currently requires DiscreteActionParser.");
  }
  action_dim_ = static_cast<int>(discrete->action_table().size());

  current_buffers_ = allocate_host_buffers(pin_host_memory);
  next_buffers_ = allocate_host_buffers(pin_host_memory);

  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  if (pin_host_memory) {
    f32 = f32.pinned_memory(true);
    i64 = i64.pinned_memory(true);
    u8 = u8.pinned_memory(true);
  }

  host_dones_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_terminated_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_truncated_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_terminal_outcome_labels_ = torch::full({static_cast<long>(total_agents_)}, 2, i64);
  host_terminal_observations_ =
      torch::zeros({static_cast<long>(total_agents_), obs_dim_}, f32);
  host_goal_positions_ = torch::zeros({static_cast<long>(total_agents_), 3}, f32);
  host_ball_proximity_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_episode_ball_touch_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_episode_ball_touch_count_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_rewards_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_gameplay_rewards_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_mechanic_rewards_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_sparse_events_ = torch::zeros(
      {static_cast<long>(total_agents_), static_cast<long>(kSparseEventChannelCount)}, u8);
  host_env_touched_ = torch::zeros({static_cast<long>(envs_.size())}, f32);
  host_env_multi_touched_ = torch::zeros({static_cast<long>(envs_.size())}, f32);
  host_bootstrap_truncated_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  agent_reward_states_.resize(total_agents_);
  env_reward_states_.resize(envs_.size());

  for (std::size_t env_idx = 0; env_idx < envs_.size(); ++env_idx) {
    assign_env(env_idx, envs_[env_idx].reset_seed);
  }
  current_buffers_.episode_starts.fill_(1.0F);
  rebuild_host_buffers(current_buffers_, nullptr);
}

void BatchedRocketSimCollector::set_self_play_assignment_fn(AssignmentFn assignment_fn) {
  assignment_fn_ = std::move(assignment_fn);
  for (std::size_t env_idx = 0; env_idx < envs_.size(); ++env_idx) {
    assign_env(env_idx, envs_[env_idx].reset_seed);
  }
  current_buffers_.episode_starts.fill_(1.0F);
  rebuild_host_buffers(current_buffers_, nullptr);
}

void BatchedRocketSimCollector::update_reward_config(const ExperimentConfig& cfg) {
  reward_engine_.update_config(cfg);
}

void BatchedRocketSimCollector::update_unlocked_mechanics(const std::vector<std::string>& mechanics) {
  reward_engine_.set_unlocked_mechanics(mechanics);
}

void BatchedRocketSimCollector::set_mode(const std::string& mode) {
  mode_ = mode;
}

const std::string& BatchedRocketSimCollector::mode() const {
  return mode_;
}

std::int8_t BatchedRocketSimCollector::mode_id() const {
  if (mode_ == "1v1") return 1;
  if (mode_ == "2v2") return 2;
  if (mode_ == "3v3") return 3;
  return 0;
}

void BatchedRocketSimCollector::reset_all(CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "reset_all");
  const auto reset_start = std::chrono::steady_clock::now();
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      envs_[env_idx].reset_seed += static_cast<std::uint64_t>(envs_.size());
      envs_[env_idx].engine->reset(envs_[env_idx].reset_seed);
    }
  });
  for (std::size_t env_idx = 0; env_idx < envs_.size(); ++env_idx) {
    assign_env(env_idx, envs_[env_idx].reset_seed);
  }
  host_dones_.zero_();
  host_terminated_.zero_();
  host_truncated_.zero_();
  host_terminal_outcome_labels_.fill_(2);
  host_terminal_observations_.zero_();
  host_goal_positions_.zero_();
  host_ball_proximity_.zero_();
  host_episode_ball_touch_.zero_();
  host_episode_ball_touch_count_.zero_();
  host_rewards_.zero_();
  host_gameplay_rewards_.zero_();
  host_mechanic_rewards_.zero_();
  host_sparse_events_.zero_();
  host_env_touched_.zero_();
  host_env_multi_touched_.zero_();
  host_bootstrap_truncated_.zero_();
  agent_reward_states_.assign(total_agents_, AgentRewardState{});
  env_reward_states_.assign(envs_.size(), EnvRewardState{});
  current_buffers_.episode_starts.fill_(1.0F);
  rebuild_host_buffers(current_buffers_, timings);
  if (timings != nullptr) {
    timings->done_reset_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reset_start).count();
  }
}

void BatchedRocketSimCollector::reset_es_episode(int update_index, int episode_index, int eval_envs_per_member, CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "reset_es_episode");
  const auto reset_start = std::chrono::steady_clock::now();
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const int local_env = static_cast<int>(env_idx) % eval_envs_per_member;
      envs_[env_idx].reset_seed = static_cast<std::uint64_t>(
          config_.env.seed + 1'000'003 + update_index * 65'537 + episode_index * 8'191 + local_env);
      envs_[env_idx].engine->reset(envs_[env_idx].reset_seed);
    }
  });
  for (std::size_t env_idx = 0; env_idx < envs_.size(); ++env_idx) {
    assign_env(env_idx, envs_[env_idx].reset_seed);
  }
  host_dones_.zero_();
  host_terminated_.zero_();
  host_truncated_.zero_();
  host_terminal_outcome_labels_.fill_(2);
  host_terminal_observations_.zero_();
  host_goal_positions_.zero_();
  host_ball_proximity_.zero_();
  host_episode_ball_touch_.zero_();
  host_episode_ball_touch_count_.zero_();
  host_rewards_.zero_();
  host_gameplay_rewards_.zero_();
  host_mechanic_rewards_.zero_();
  host_sparse_events_.zero_();
  host_env_touched_.zero_();
  host_env_multi_touched_.zero_();
  host_bootstrap_truncated_.zero_();
  agent_reward_states_.assign(total_agents_, AgentRewardState{});
  env_reward_states_.assign(envs_.size(), EnvRewardState{});
  current_buffers_.episode_starts.fill_(1.0F);
  rebuild_host_buffers(current_buffers_, timings);
  if (timings != nullptr) {
    timings->done_reset_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reset_start).count();
  }
}

std::size_t BatchedRocketSimCollector::num_envs() const {
  return envs_.size();
}

std::size_t BatchedRocketSimCollector::total_agents() const {
  return total_agents_;
}

int BatchedRocketSimCollector::obs_dim() const {
  return obs_dim_;
}

int BatchedRocketSimCollector::action_dim() const {
  return action_dim_;
}

BatchedRocketSimCollector::HostBuffers BatchedRocketSimCollector::allocate_host_buffers(bool pin_host_memory) const {
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  if (pin_host_memory) {
    f32 = f32.pinned_memory(true);
    u8 = u8.pinned_memory(true);
    i64 = i64.pinned_memory(true);
  }

  return HostBuffers{
      .obs = torch::empty({static_cast<long>(total_agents_), obs_dim_}, f32),
      .action_masks = torch::empty({static_cast<long>(total_agents_), action_dim_}, u8),
      .learner_active = torch::ones({static_cast<long>(total_agents_)}, f32),
      .snapshot_ids = torch::full({static_cast<long>(total_agents_)}, -1, i64),
      .episode_starts = torch::zeros({static_cast<long>(total_agents_)}, f32),
  };
}

void BatchedRocketSimCollector::assign_env(std::size_t env_idx, std::uint64_t seed) {
  if (assignment_fn_) {
    envs_[env_idx].assignment = assignment_fn_(env_idx, seed);
  } else {
    envs_[env_idx].assignment = {};
  }
  refresh_obs_car_order(env_idx, seed);
}

void BatchedRocketSimCollector::refresh_obs_car_order(std::size_t env_idx, std::uint64_t seed) {
  auto& order = envs_[env_idx].obs_car_order;
  order.resize(envs_[env_idx].engine->num_agents());
  std::iota(order.begin(), order.end(), AgentId{0});
  if (order.size() <= 2) return;

  std::mt19937_64 rng(mix_obs_order_seed(seed, env_idx));
  std::shuffle(order.begin(), order.end(), rng);
}

void BatchedRocketSimCollector::build_env_obs_batch(std::size_t env_idx, const EnvState& state, std::span<float> out) const {
  if (const auto* pulsar_obs_builder = dynamic_cast<const PulsarObsBuilder*>(obs_builder_.get())) {
    pulsar_obs_builder->build_obs_batch(state, out, envs_[env_idx].obs_car_order);
    return;
  }
  obs_builder_->build_obs_batch(state, out);
}

void BatchedRocketSimCollector::rebuild_host_buffers(HostBuffers& buffers, CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "rebuild_host_buffers");
  const auto obs_start = std::chrono::steady_clock::now();
  float* obs_ptr = buffers.obs.data_ptr<float>();
  const std::size_t obs_stride = static_cast<std::size_t>(obs_dim_);
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_offset = agent_offsets_[env_idx];
      const std::size_t count = envs_[env_idx].engine->num_agents();
      build_env_obs_batch(
          env_idx,
          envs_[env_idx].engine->state(),
          std::span<float>(
              obs_ptr + static_cast<std::ptrdiff_t>(agent_offset * obs_stride),
              count * obs_stride));
    }
  });
  if (timings != nullptr) {
    timings->obs_build_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - obs_start).count();
  }

  const auto mask_start = std::chrono::steady_clock::now();
  std::uint8_t* masks_ptr = buffers.action_masks.data_ptr<std::uint8_t>();
  float* learner_ptr = buffers.learner_active.data_ptr<float>();
  std::int64_t* snapshot_ptr = buffers.snapshot_ids.data_ptr<std::int64_t>();
  const std::size_t action_stride = static_cast<std::size_t>(action_dim_);
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const EnvState& state = envs_[env_idx].engine->state();
      const std::size_t agent_offset = agent_offsets_[env_idx];
      action_parser_->build_action_mask_batch(
          state,
          std::span<std::uint8_t>(
              masks_ptr + static_cast<std::ptrdiff_t>(agent_offset * action_stride),
              state.cars.size() * action_stride));

      for (std::size_t local_idx = 0; local_idx < state.cars.size(); ++local_idx) {
        const std::size_t global_idx = agent_offset + local_idx;
        const SelfPlayAssignment& assignment = envs_[env_idx].assignment;
        const bool self_play =
            assignment.enabled && assignment.snapshot_index >= 0;
        const bool learner =
            !self_play || state.cars[local_idx].team == assignment.learner_team;
        learner_ptr[global_idx] = learner ? 1.0F : 0.0F;
        snapshot_ptr[global_idx] = (self_play && !learner)
            ? static_cast<std::int64_t>(assignment.snapshot_index)
            : -1;
      }
    }
  });
  if (timings != nullptr) {
    timings->mask_build_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - mask_start).count();
  }
}

void BatchedRocketSimCollector::rebuild_next_buffers(CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "rebuild_next_buffers");
  next_buffers_.episode_starts.copy_(host_dones_);
  rebuild_host_buffers(next_buffers_, timings);
}

void BatchedRocketSimCollector::collect_live_self_play_outcomes() {
  last_self_play_outcomes_.clear();
  const auto* dones_ptr = host_dones_.data_ptr<float>();
  const auto* labels_ptr = host_terminal_outcome_labels_.data_ptr<std::int64_t>();
  const auto* learner_ptr = current_buffers_.learner_active.data_ptr<float>();

  for (std::size_t env_idx = 0; env_idx < envs_.size(); ++env_idx) {
    const SelfPlayAssignment& assignment = envs_[env_idx].assignment;
    if (!assignment.enabled || assignment.snapshot_index < 0) {
      continue;
    }

    const std::size_t agent_begin = agent_offsets_[env_idx];
    const std::size_t agent_end = agent_offsets_[env_idx + 1];
    bool env_done = false;
    bool has_learner_terminal = false;
    int learner_result = 0;
    for (std::size_t gi = agent_begin; gi < agent_end; ++gi) {
      if (dones_ptr[gi] <= 0.5F) {
        continue;
      }
      env_done = true;
      if (learner_ptr[gi] <= 0.5F) {
        continue;
      }
      has_learner_terminal = true;
      const std::int64_t label = labels_ptr[gi];
      if (label == 0) {
        learner_result = 1;
        break;
      }
      if (label == 1) {
        learner_result = -1;
        break;
      }
    }
    if (env_done && has_learner_terminal) {
      last_self_play_outcomes_.push_back(SelfPlayLiveOutcome{
          .snapshot_index = assignment.snapshot_index,
          .learner_result = learner_result,
      });
    }
  }
}

void BatchedRocketSimCollector::finalize_step(CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "finalize_step");
  const auto done_reset_start = std::chrono::steady_clock::now();
  float* dones_ptr = host_dones_.data_ptr<float>();
  float* terminated_ptr = host_terminated_.data_ptr<float>();
  float* truncated_ptr = host_truncated_.data_ptr<float>();
  std::int64_t* labels_ptr = host_terminal_outcome_labels_.data_ptr<std::int64_t>();
  float* terminal_obs_ptr = host_terminal_observations_.data_ptr<float>();
  float* goal_pos_ptr = host_goal_positions_.data_ptr<float>();
  float* ball_touch_ptr = host_ball_proximity_.data_ptr<float>();
  float* episode_touch_ptr = host_episode_ball_touch_.data_ptr<float>();
  float* episode_touch_count_ptr = host_episode_ball_touch_count_.data_ptr<float>();
  float* reward_ptr = host_rewards_.data_ptr<float>();
  float* gp_reward_ptr = host_gameplay_rewards_.data_ptr<float>();
  float* mech_reward_ptr = host_mechanic_rewards_.data_ptr<float>();
  float* bootstrap_truncated_ptr = host_bootstrap_truncated_.data_ptr<float>();
  const std::size_t obs_stride = static_cast<std::size_t>(obs_dim_);
  const bool has_team_spirit = reward_engine_.has_team_spirit();
  host_dones_.zero_();
  host_terminated_.zero_();
  host_truncated_.zero_();
  host_terminal_outcome_labels_.fill_(2);
  host_terminal_observations_.zero_();
  host_goal_positions_.zero_();
  host_ball_proximity_.zero_();
  host_rewards_.zero_();
  host_gameplay_rewards_.zero_();
  host_mechanic_rewards_.zero_();
  host_sparse_events_.zero_();
  host_env_touched_.zero_();
  host_env_multi_touched_.zero_();
  host_bootstrap_truncated_.zero_();

  float* host_env_touched_ptr = host_env_touched_.data_ptr<float>();
  float* host_env_multi_touched_ptr = host_env_multi_touched_.data_ptr<float>();
  const float* learner_ptr = current_buffers_.learner_active.data_ptr<float>();
  std::uint8_t* sparse_events_ptr = host_sparse_events_.data_ptr<std::uint8_t>();

  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_begin = agent_offsets_[env_idx];
      const std::size_t agent_end = agent_offsets_[env_idx + 1];
      const std::size_t count = agent_end - agent_begin;
      const EnvState& current_state = envs_[env_idx].engine->state();
      std::vector<AgentRewardState> previous_agent_states(count);
      for (std::size_t idx = 0; idx < count; ++idx) {
        previous_agent_states[idx] = agent_reward_states_[agent_begin + idx];
      }

      done_condition_->is_done_into(
          current_state,
          current_state.tick,
          envs_[env_idx].terminated_scratch,
          envs_[env_idx].truncated_scratch);

      bool reset_needed = false;
      const bool goal_scored = current_state.goal_scored;
      const Team scoring_team = current_state.last_scoring_team;
      float team_touch_count[2] = {0.0F, 0.0F};
      for (std::size_t idx = 0; idx < count; ++idx) {
        const CarState& car = current_state.cars[idx];
        const BallState& ball = current_state.ball;

        const float dx = car.position.x - ball.position.x;
        const float dy = car.position.y - ball.position.y;
        const float dz = car.position.z - ball.position.z;
        const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float step_proximity = (dist < 300.0F) ? 1.0F : 0.0F;

        const std::size_t global_idx = agent_begin + idx;
        const bool touch_edge = car.ball_touched && !previous_agent_states[idx].prev_ball_touched;
        float& accumulated = episode_touch_ptr[global_idx];
        accumulated = std::max(accumulated, touch_edge ? 1.0F : 0.0F);
        if (touch_edge) {
          episode_touch_count_ptr[global_idx] += 1.0F;
        }
        ball_touch_ptr[global_idx] = step_proximity;

        const int team_index = (car.team == Team::Blue) ? 0 : 1;
        team_touch_count[team_index] += episode_touch_count_ptr[global_idx];

        const bool is_terminated = envs_[env_idx].terminated_scratch[idx] != 0;
        const bool is_truncated = envs_[env_idx].truncated_scratch[idx] != 0;
        const bool done = is_terminated || is_truncated;
        dones_ptr[global_idx] = done ? 1.0F : 0.0F;
        terminated_ptr[global_idx] = is_terminated ? 1.0F : 0.0F;
        truncated_ptr[global_idx] = is_truncated ? 1.0F : 0.0F;
        bootstrap_truncated_ptr[global_idx] = (is_truncated && !is_terminated) ? 1.0F : 0.0F;
        reset_needed = reset_needed || done;
      }

      for (std::size_t idx = 0; idx < count; ++idx) {
        const CarState& car = current_state.cars[idx];
        const std::size_t global_idx = agent_begin + idx;
        const bool is_terminated = envs_[env_idx].terminated_scratch[idx] != 0;
        const bool is_truncated = envs_[env_idx].truncated_scratch[idx] != 0;
        const bool done = is_terminated || is_truncated;
        const int team_index = (car.team == Team::Blue) ? 0 : 1;

        const int label = done
            ? (goal_scored
                ? (car.team == scoring_team ? 0 : 1)
                : (team_touch_count[team_index] > 0.5F ? 2 : 3))
            : -1;
        if (done) {
          labels_ptr[global_idx] = label;
        }

        reward_engine_.detect_sparse_events(
            current_state.tick,
            car,
            current_state,
            idx,
            std::span<const AgentRewardState>(previous_agent_states.data(), previous_agent_states.size()),
            agent_reward_states_[global_idx],
            env_reward_states_[env_idx],
            done,
            label,
            sparse_events_ptr + global_idx * kSparseEventChannelCount);

        RewardBreakdown breakdown = reward_engine_.compute(
            current_state.tick, car, current_state,
            static_cast<int>(config_.env.team_size),
            agent_reward_states_[global_idx],
            env_reward_states_[env_idx],
            done, label);
        reward_ptr[global_idx] = breakdown.total;
        gp_reward_ptr[global_idx] = breakdown.gameplay;
        mech_reward_ptr[global_idx] = breakdown.mechanic;

        float goal_pos[3];
        compute_goal_position(current_state, config_.goal_mapping, goal_pos);
        if (config_.env.obs_x_mirror) {
          const bool inverted = car.team == Team::Orange;
          const float self_x = inverted ? -car.position.x : car.position.x;
          if (self_x > 0.0F) {
            goal_pos[0] = -goal_pos[0];
          }
        }
        const int pos_offset = static_cast<int>(global_idx) * 3;
        goal_pos_ptr[pos_offset + 0] = goal_pos[0];
        goal_pos_ptr[pos_offset + 1] = goal_pos[1];
        goal_pos_ptr[pos_offset + 2] = goal_pos[2];
      }

      if (has_team_spirit) {
        const float ts = reward_engine_.outcome_team_spirit();
        float team_reward[2] = {0.0F, 0.0F};
        float team_gp_reward[2] = {0.0F, 0.0F};
        float team_mech_reward[2] = {0.0F, 0.0F};
        for (std::size_t idx = 0; idx < count; ++idx) {
          const int team_index = (current_state.cars[idx].team == Team::Blue) ? 0 : 1;
          team_reward[team_index] += reward_ptr[agent_begin + idx];
          team_gp_reward[team_index] += gp_reward_ptr[agent_begin + idx];
          team_mech_reward[team_index] += mech_reward_ptr[agent_begin + idx];
        }
        for (std::size_t idx = 0; idx < count; ++idx) {
          const int team_index = (current_state.cars[idx].team == Team::Blue) ? 0 : 1;
          const std::size_t global_idx = agent_begin + idx;
          reward_ptr[global_idx] += ts * (team_reward[team_index] - reward_ptr[global_idx]);
          gp_reward_ptr[global_idx] += ts * (team_gp_reward[team_index] - gp_reward_ptr[global_idx]);
          mech_reward_ptr[global_idx] += ts * (team_mech_reward[team_index] - mech_reward_ptr[global_idx]);
        }
      }

      if (reset_needed) {
        build_env_obs_batch(
            env_idx,
            current_state,
            std::span<float>(
                terminal_obs_ptr + static_cast<std::ptrdiff_t>(agent_begin * obs_stride),
                count * obs_stride));
        envs_[env_idx].reset_seed += static_cast<std::uint64_t>(envs_.size());
        envs_[env_idx].engine->reset(envs_[env_idx].reset_seed);
        assign_env(env_idx, envs_[env_idx].reset_seed);
        env_reward_states_[env_idx] = EnvRewardState{};
        float learner_touch_count = 0.0F;
        for (std::size_t idx = 0; idx < count; ++idx) {
          const std::size_t global_idx = agent_begin + idx;
          if (learner_ptr[global_idx] > 0.5F) {
            learner_touch_count += episode_touch_count_ptr[global_idx];
          }
        }
        const bool learner_touched = learner_touch_count >= 1.0F;
        const bool learner_multi_touched = learner_touch_count >= 2.0F;
        if (learner_touched) {
          host_env_touched_ptr[env_idx] = 1.0F;
        }
        if (learner_multi_touched) {
          host_env_multi_touched_ptr[env_idx] = 1.0F;
        }
        for (std::size_t idx = 0; idx < count; ++idx) {
          episode_touch_ptr[agent_begin + idx] = 0.0F;
          episode_touch_count_ptr[agent_begin + idx] = 0.0F;
          agent_reward_states_[agent_begin + idx] = AgentRewardState{};
        }
      }
    }
  });
  if (timings != nullptr) {
    timings->done_reset_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - done_reset_start).count();
  }

  collect_live_self_play_outcomes();
  rebuild_next_buffers(timings);
  std::swap(current_buffers_, next_buffers_);
}

void BatchedRocketSimCollector::step(std::span<const ControllerState> actions, CollectorTimings* timings) {
  step_after_physics_prefix(actions, 0, timings);
}

void BatchedRocketSimCollector::step_after_physics_prefix(
    std::span<const ControllerState> actions,
    int prefix_tick_count,
    CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "step_controller");
  if (actions.size() != total_agents_) {
    throw std::invalid_argument("BatchedRocketSimCollector::step action span has incorrect size.");
  }
  const int prefix_ticks = bounded_physics_prefix_ticks(prefix_tick_count);
  const int remaining_ticks = config_.env.tick_skip - prefix_ticks;

  const auto env_step_start = std::chrono::steady_clock::now();
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_begin = agent_offsets_[env_idx];
      const std::size_t agent_end = agent_offsets_[env_idx + 1];
      auto& engine = *envs_[env_idx].engine;
      const std::size_t count = agent_end - agent_begin;
      auto& action_scratch = envs_[env_idx].action_scratch;
      const EnvState& current_state = engine.state();
      for (std::size_t idx = 0; idx < count; ++idx) {
        action_scratch[idx] = actions[agent_begin + idx];
        if (config_.env.obs_x_mirror) {
          const CarState& car = current_state.cars[idx];
          const bool inverted = car.team == Team::Orange;
          const float self_x = inverted ? -car.position.x : car.position.x;
          if (self_x > 0.0F) {
            action_scratch[idx].steer = -action_scratch[idx].steer;
            action_scratch[idx].yaw = -action_scratch[idx].yaw;
            action_scratch[idx].roll = -action_scratch[idx].roll;
          }
        }
      }
      if (remaining_ticks > 0) {
        engine.apply_controls(action_scratch);
        engine.step_physics(remaining_ticks);
      } else if (prefix_ticks == 0) {
        engine.step_inplace(action_scratch);
      }
    }
  });
  if (timings != nullptr) {
    timings->env_step_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - env_step_start).count();
  }
  finalize_step(timings);
}

void BatchedRocketSimCollector::step_physics_only(int tick_count, CollectorTimings* timings) {
  if (tick_count <= 0) {
    return;
  }
  const auto env_step_start = std::chrono::steady_clock::now();
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      envs_[env_idx].engine->apply_controls(envs_[env_idx].action_scratch);
      envs_[env_idx].engine->step_physics(tick_count);
    }
  });
  if (timings != nullptr) {
    timings->env_step_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - env_step_start).count();
  }
}

void BatchedRocketSimCollector::step(std::span<const std::int64_t> action_indices, CollectorTimings* timings) {
  step_after_physics_prefix(action_indices, 0, timings);
}

void BatchedRocketSimCollector::step_after_physics_prefix(
    std::span<const std::int64_t> action_indices,
    int prefix_tick_count,
    CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "step_discrete_fused");
  if (action_indices.size() != total_agents_) {
    throw std::invalid_argument("BatchedRocketSimCollector::step action span has incorrect size.");
  }
  const int prefix_ticks = bounded_physics_prefix_ticks(prefix_tick_count);
  const int remaining_ticks = config_.env.tick_skip - prefix_ticks;

  const auto step_start = std::chrono::steady_clock::now();

  // Reset step output
  last_step_output_ = StepOutput{};
  last_self_play_outcomes_.clear();
  auto& so = last_step_output_;

  // Output tensor pointers
  float* dones_ptr = host_dones_.data_ptr<float>();
  float* terminated_ptr = host_terminated_.data_ptr<float>();
  float* truncated_ptr = host_truncated_.data_ptr<float>();
  std::int64_t* labels_ptr = host_terminal_outcome_labels_.data_ptr<std::int64_t>();
  float* terminal_obs_ptr = host_terminal_observations_.data_ptr<float>();
  float* goal_pos_ptr = host_goal_positions_.data_ptr<float>();
  float* ball_touch_ptr = host_ball_proximity_.data_ptr<float>();
  float* episode_touch_ptr = host_episode_ball_touch_.data_ptr<float>();
  float* episode_touch_count_ptr = host_episode_ball_touch_count_.data_ptr<float>();
  float* reward_ptr = host_rewards_.data_ptr<float>();
  float* gp_reward_ptr = host_gameplay_rewards_.data_ptr<float>();
  float* mech_reward_ptr = host_mechanic_rewards_.data_ptr<float>();
  std::uint8_t* sparse_events_ptr = host_sparse_events_.data_ptr<std::uint8_t>();
  float* env_touch_ptr = host_env_touched_.data_ptr<float>();
  float* env_multi_touch_ptr = host_env_multi_touched_.data_ptr<float>();
  float* bootstrap_truncated_ptr = host_bootstrap_truncated_.data_ptr<float>();

  // Next buffer pointers. `episode_starts` is copied after the step loop,
  // once host_dones_ contains this step's freshly-computed reset flags.
  float* next_obs_ptr = next_buffers_.obs.data_ptr<float>();
  std::uint8_t* next_masks_ptr = next_buffers_.action_masks.data_ptr<std::uint8_t>();
  float* next_learner_ptr = next_buffers_.learner_active.data_ptr<float>();
  std::int64_t* next_snapshot_ptr = next_buffers_.snapshot_ids.data_ptr<std::int64_t>();

  const std::size_t obs_stride = static_cast<std::size_t>(obs_dim_);
  const std::size_t action_stride = static_cast<std::size_t>(action_dim_);
  const float* ler_ptr = current_buffers_.learner_active.data_ptr<float>();

  const bool has_team_spirit = reward_engine_.has_team_spirit();

  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_begin = agent_offsets_[env_idx];
      const std::size_t agent_end = agent_offsets_[env_idx + 1];
      const std::size_t count = agent_end - agent_begin;

      // Parse actions + step RocketSim
      action_parser_->parse_actions_into(
          std::span<const std::int64_t>(
              action_indices.data() + static_cast<std::ptrdiff_t>(agent_begin), count),
          envs_[env_idx].action_scratch);
      auto& engine = *envs_[env_idx].engine;
      if (config_.env.obs_x_mirror) {
        const EnvState& pre_step_state = engine.state();
        for (std::size_t idx = 0; idx < count; ++idx) {
          const CarState& car = pre_step_state.cars[idx];
          const bool inverted = car.team == Team::Orange;
          const float self_x = inverted ? -car.position.x : car.position.x;
          if (self_x > 0.0F) {
            auto& act = envs_[env_idx].action_scratch[idx];
            act.steer = -act.steer;
            act.yaw = -act.yaw;
            act.roll = -act.roll;
          }
        }
      }
      if (remaining_ticks > 0) {
        engine.apply_controls(envs_[env_idx].action_scratch);
        engine.step_physics(remaining_ticks);
      } else if (prefix_ticks == 0) {
        engine.step_inplace(envs_[env_idx].action_scratch);
      }

      const EnvState& current_state = envs_[env_idx].engine->state();
      std::vector<AgentRewardState> previous_agent_states(count);
      for (std::size_t idx = 0; idx < count; ++idx) {
        previous_agent_states[idx] = agent_reward_states_[agent_begin + idx];
      }

      // Done/truncated check
      done_condition_->is_done_into(
          current_state, current_state.tick,
          envs_[env_idx].terminated_scratch,
          envs_[env_idx].truncated_scratch);

      bool reset_needed = false;
      const bool goal_scored = current_state.goal_scored;
      const Team scoring_team = current_state.last_scoring_team;
      float team_touch_count[2] = {0.0F, 0.0F};

      // Per-agent: proximity, touch, done flags
      for (std::size_t idx = 0; idx < count; ++idx) {
        const CarState& car = current_state.cars[idx];
        const BallState& ball = current_state.ball;
        const std::size_t gi = agent_begin + idx;

        const float dx = car.position.x - ball.position.x;
        const float dy = car.position.y - ball.position.y;
        const float dz = car.position.z - ball.position.z;
        ball_touch_ptr[gi] = (dx * dx + dy * dy + dz * dz < 90000.0F) ? 1.0F : 0.0F;

        const bool touch_edge = car.ball_touched && !previous_agent_states[idx].prev_ball_touched;
        float& acc = episode_touch_ptr[gi];
        acc = std::max(acc, touch_edge ? 1.0F : 0.0F);
        if (touch_edge) episode_touch_count_ptr[gi] += 1.0F;

        const int ti = (car.team == Team::Blue) ? 0 : 1;
        team_touch_count[ti] += episode_touch_count_ptr[gi];

        const bool is_term = envs_[env_idx].terminated_scratch[idx] != 0;
        const bool is_trunc = envs_[env_idx].truncated_scratch[idx] != 0;
        const bool done = is_term || is_trunc;
        dones_ptr[gi] = done ? 1.0F : 0.0F;
        terminated_ptr[gi] = is_term ? 1.0F : 0.0F;
        truncated_ptr[gi] = is_trunc ? 1.0F : 0.0F;
        bootstrap_truncated_ptr[gi] = (is_trunc && !is_term) ? 1.0F : 0.0F;
        reset_needed = reset_needed || done;
      }

      // Per-agent: labels, rewards, goal pos
      for (std::size_t idx = 0; idx < count; ++idx) {
        const CarState& car = current_state.cars[idx];
        const std::size_t gi = agent_begin + idx;
        const bool is_term = envs_[env_idx].terminated_scratch[idx] != 0;
        const bool is_trunc = envs_[env_idx].truncated_scratch[idx] != 0;
        const bool done = is_term || is_trunc;
        const int ti = (car.team == Team::Blue) ? 0 : 1;

        const int label = done
            ? (goal_scored
                ? (car.team == scoring_team ? 0 : 1)
                : (team_touch_count[ti] > 0.5F ? 2 : 3))
            : -1;
        labels_ptr[gi] = static_cast<std::int64_t>(label);

        reward_engine_.detect_sparse_events(
            current_state.tick,
            car,
            current_state,
            idx,
            std::span<const AgentRewardState>(previous_agent_states.data(), previous_agent_states.size()),
            agent_reward_states_[gi],
            env_reward_states_[env_idx],
            done,
            label,
            sparse_events_ptr + gi * kSparseEventChannelCount);

        RewardBreakdown bd = reward_engine_.compute(
            current_state.tick, car, current_state,
            static_cast<int>(config_.env.team_size),
            agent_reward_states_[gi], env_reward_states_[env_idx],
            done, label);
        reward_ptr[gi] = bd.total;
        gp_reward_ptr[gi] = bd.gameplay;
        mech_reward_ptr[gi] = bd.mechanic;

        float goal_pos[3];
        compute_goal_position(current_state, config_.goal_mapping, goal_pos);
        if (config_.env.obs_x_mirror) {
          const bool inverted = car.team == Team::Orange;
          const float self_x = inverted ? -car.position.x : car.position.x;
          if (self_x > 0.0F) {
            goal_pos[0] = -goal_pos[0];
          }
        }
        const int po = static_cast<int>(gi) * 3;
        goal_pos_ptr[po] = goal_pos[0];
        goal_pos_ptr[po + 1] = goal_pos[1];
        goal_pos_ptr[po + 2] = goal_pos[2];
      }

      // Team spirit: each agent's reward gets a fraction of teammates' rewards
      if (has_team_spirit) {
        const float ts = reward_engine_.outcome_team_spirit();
        float team_reward[2] = {0.0F, 0.0F};
        float team_gp_reward[2] = {0.0F, 0.0F};
        float team_mech_reward[2] = {0.0F, 0.0F};
        for (std::size_t idx = 0; idx < count; ++idx) {
          const int ti = (current_state.cars[idx].team == Team::Blue) ? 0 : 1;
          team_reward[ti] += reward_ptr[agent_begin + idx];
          team_gp_reward[ti] += gp_reward_ptr[agent_begin + idx];
          team_mech_reward[ti] += mech_reward_ptr[agent_begin + idx];
        }
        for (std::size_t idx = 0; idx < count; ++idx) {
          const int ti = (current_state.cars[idx].team == Team::Blue) ? 0 : 1;
          reward_ptr[agent_begin + idx] += ts * (team_reward[ti] - reward_ptr[agent_begin + idx]);
          gp_reward_ptr[agent_begin + idx] += ts * (team_gp_reward[ti] - gp_reward_ptr[agent_begin + idx]);
          mech_reward_ptr[agent_begin + idx] += ts * (team_mech_reward[ti] - mech_reward_ptr[agent_begin + idx]);
        }
      }



      // Reset handling
      if (reset_needed) {
        build_env_obs_batch(env_idx, current_state,
            std::span<float>(terminal_obs_ptr + static_cast<std::ptrdiff_t>(agent_begin * obs_stride),
                             count * obs_stride));
        envs_[env_idx].reset_seed += static_cast<std::uint64_t>(envs_.size());
        envs_[env_idx].engine->reset(envs_[env_idx].reset_seed);
        assign_env(env_idx, envs_[env_idx].reset_seed);
        env_reward_states_[env_idx] = EnvRewardState{};

        float ltc = 0.0F;
        for (std::size_t i = 0; i < count; ++i)
          if (ler_ptr[agent_begin + i] > 0.5F) ltc += episode_touch_count_ptr[agent_begin + i];
        env_touch_ptr[env_idx] = (ltc >= 1.0F) ? 1.0F : 0.0F;
        env_multi_touch_ptr[env_idx] = (ltc >= 2.0F) ? 1.0F : 0.0F;

        for (std::size_t i = 0; i < count; ++i) {
          episode_touch_ptr[agent_begin + i] = 0.0F;
          episode_touch_count_ptr[agent_begin + i] = 0.0F;
          agent_reward_states_[agent_begin + i] = AgentRewardState{};
        }
      } else {
        env_touch_ptr[env_idx] = 0.0F;
        env_multi_touch_ptr[env_idx] = 0.0F;
      }

      // Build next obs
      const EnvState& next_state = envs_[env_idx].engine->state();
      build_env_obs_batch(env_idx, next_state,
          std::span<float>(next_obs_ptr + static_cast<std::ptrdiff_t>(agent_begin * obs_stride),
                           count * obs_stride));

      // Build next action masks + learner/snapshot flags
      action_parser_->build_action_mask_batch(next_state,
          std::span<std::uint8_t>(next_masks_ptr + static_cast<std::ptrdiff_t>(agent_begin * action_stride),
                                   count * action_stride));

      for (std::size_t i = 0; i < count; ++i) {
        const std::size_t gi = agent_begin + i;
        const SelfPlayAssignment& assignment = envs_[env_idx].assignment;
        const bool self_play =
            assignment.enabled && assignment.snapshot_index >= 0;
        const bool learner =
            !self_play || next_state.cars[i].team == assignment.learner_team;
        next_learner_ptr[gi] = learner ? 1.0F : 0.0F;
        next_snapshot_ptr[gi] = (self_play && !learner)
            ? static_cast<std::int64_t>(assignment.snapshot_index)
            : -1;
      }
    }
  });

  collect_live_self_play_outcomes();
  const torch::Tensor step_learner_active = current_buffers_.learner_active;

  // Mark the next observation as an episode start exactly when the just-taken
  // step ended an episode. This must happen after the loop fills host_dones_.
  next_buffers_.episode_starts.copy_(host_dones_);

  // Swap buffers
  std::swap(current_buffers_, next_buffers_);

  // Compute scalar step output
  const auto* la_ptr = step_learner_active.data_ptr<float>();
  const float* env_t_ptr = host_env_touched_.data_ptr<float>();
  const float* env_mt_ptr = host_env_multi_touched_.data_ptr<float>();
  const int64_t n_agents = static_cast<int64_t>(total_agents_);
  const int64_t n_envs = static_cast<int64_t>(envs_.size());
  const int64_t ape = (n_envs > 0) ? (n_agents / n_envs) : 2;

  for (int64_t gi = 0; gi < n_agents; ++gi) {
    so.total_steps++;
    if (la_ptr[gi] > 0.5F) {
      so.total_learner_steps++;
      so.total_reward += static_cast<double>(reward_ptr[gi]);
      so.total_gameplay_reward += static_cast<double>(gp_reward_ptr[gi]);
      so.total_mechanic_reward += static_cast<double>(mech_reward_ptr[gi]);
    }
    float gn = std::sqrt(goal_pos_ptr[gi * 3] * goal_pos_ptr[gi * 3] +
                         goal_pos_ptr[gi * 3 + 1] * goal_pos_ptr[gi * 3 + 1] +
                         goal_pos_ptr[gi * 3 + 2] * goal_pos_ptr[gi * 3 + 2]);
    so.goal_distance_sum += static_cast<double>(gn);
    so.goal_distance_count++;
    if (gn < static_cast<float>(so.min_goal_distance)) so.min_goal_distance = static_cast<double>(gn);
    so.ball_prox_steps += (ball_touch_ptr[gi] > 0.5F) ? 1 : 0;
    so.ball_prox_denom++;
  }

  for (int64_t ei = 0; ei < n_envs; ++ei) {
    const int64_t ea = ei * ape;
    const int64_t ee = std::min<int64_t>(ea + ape, n_agents);
    bool env_done = false, env_scored = false, env_gs = false, env_gc = false;
    for (int64_t gi = ea; gi < ee; ++gi) {
      if (dones_ptr[gi] > 0.5F) {
        env_done = true;
        if (labels_ptr[gi] == 0) { env_scored = true; env_gs = true; }
        if (labels_ptr[gi] == 1) env_gc = true;
      }
    }
    if (env_gs) so.goals_scored++;
    if (env_gc) so.goals_conceded++;
    if (env_done) {
      so.completed_episodes++;
      if (env_scored) so.scored_episodes++;
      if (env_t_ptr[ei] > 0.5F) so.touched_episodes++;
      if (env_mt_ptr[ei] > 0.5F) so.multi_touched_episodes++;
    }
  }

  if (timings != nullptr) {
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    timings->env_step_seconds += s;
    timings->obs_build_seconds += s * 0.05;
    timings->mask_build_seconds += s * 0.03;
    timings->done_reset_seconds += s * 0.02;
  }
}

int BatchedRocketSimCollector::bounded_physics_prefix_ticks(int prefix_tick_count) const {
  if (prefix_tick_count <= 0 || config_.env.tick_skip <= 0) {
    return 0;
  }
  return std::min(prefix_tick_count, config_.env.tick_skip);
}

const BatchedRocketSimCollector::StepOutput& BatchedRocketSimCollector::last_step_output() const {
  return last_step_output_;
}

const std::vector<SelfPlayLiveOutcome>& BatchedRocketSimCollector::last_self_play_outcomes() const {
  return last_self_play_outcomes_;
}

const torch::Tensor& BatchedRocketSimCollector::host_observations() const {
  return current_buffers_.obs;
}

const torch::Tensor& BatchedRocketSimCollector::host_action_masks() const {
  return current_buffers_.action_masks;
}

const torch::Tensor& BatchedRocketSimCollector::host_learner_active() const {
  return current_buffers_.learner_active;
}

const torch::Tensor& BatchedRocketSimCollector::host_snapshot_ids() const {
  return current_buffers_.snapshot_ids;
}

const torch::Tensor& BatchedRocketSimCollector::host_episode_starts() const {
  return current_buffers_.episode_starts;
}

const torch::Tensor& BatchedRocketSimCollector::host_dones() const {
  return host_dones_;
}

const torch::Tensor& BatchedRocketSimCollector::host_terminated() const {
  return host_terminated_;
}

const torch::Tensor& BatchedRocketSimCollector::host_truncated() const {
  return host_truncated_;
}

const torch::Tensor& BatchedRocketSimCollector::host_bootstrap_truncated() const {
  return host_bootstrap_truncated_;
}

const torch::Tensor& BatchedRocketSimCollector::host_terminal_outcome_labels() const {
  return host_terminal_outcome_labels_;
}

const torch::Tensor& BatchedRocketSimCollector::host_terminal_observations() const {
  return host_terminal_observations_;
}

const torch::Tensor& BatchedRocketSimCollector::host_goal_positions() const {
  return host_goal_positions_;
}

const torch::Tensor& BatchedRocketSimCollector::host_ball_proximity() const {
  return host_ball_proximity_;
}

const torch::Tensor& BatchedRocketSimCollector::host_episode_ball_touch() const {
  return host_episode_ball_touch_;
}

const torch::Tensor& BatchedRocketSimCollector::host_episode_ball_touch_count() const {
  return host_episode_ball_touch_count_;
}

const torch::Tensor& BatchedRocketSimCollector::host_rewards() const {
  return host_rewards_;
}

const torch::Tensor& BatchedRocketSimCollector::host_gameplay_rewards() const {
  return host_gameplay_rewards_;
}

const torch::Tensor& BatchedRocketSimCollector::host_mechanic_rewards() const {
  return host_mechanic_rewards_;
}

const torch::Tensor& BatchedRocketSimCollector::host_sparse_events() const {
  return host_sparse_events_;
}

const torch::Tensor& BatchedRocketSimCollector::host_env_touched() const {
  return host_env_touched_;
}

const torch::Tensor& BatchedRocketSimCollector::host_env_multi_touched() const {
  return host_env_multi_touched_;
}

}  // namespace pulsar

#endif
