#include "pulsar/training/batched_rocketsim_collector.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
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
  if (pin_host_memory) {
    f32 = f32.pinned_memory(true);
    i64 = i64.pinned_memory(true);
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
  current_buffers_.episode_starts.zero_();
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
      obs_builder_->build_obs_batch(
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
        const Team team = state.cars[local_idx].team;
        const bool learner =
            !envs_[env_idx].assignment.enabled || team == envs_[env_idx].assignment.learner_team;
        learner_ptr[global_idx] = learner ? 1.0F : 0.0F;
        snapshot_ptr[global_idx] =
            learner ? -1 : static_cast<std::int64_t>(envs_[env_idx].assignment.snapshot_index);
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
  const std::size_t obs_stride = static_cast<std::size_t>(obs_dim_);
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
  host_env_touched_.zero_();
  host_env_multi_touched_.zero_();
  host_bootstrap_truncated_.zero_();

  float* host_env_touched_ptr = host_env_touched_.data_ptr<float>();
  float* host_env_multi_touched_ptr = host_env_multi_touched_.data_ptr<float>();
  const float* learner_ptr = current_buffers_.learner_active.data_ptr<float>();

  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_begin = agent_offsets_[env_idx];
      const std::size_t agent_end = agent_offsets_[env_idx + 1];
      const std::size_t count = agent_end - agent_begin;
      const EnvState& current_state = envs_[env_idx].engine->state();

      done_condition_->is_done_into(
          current_state,
          current_state.tick,
          envs_[env_idx].terminated_scratch,
          envs_[env_idx].truncated_scratch);

      bool reset_needed = false;
      const bool goal_scored = current_state.goal_scored;
      const Team scoring_team = current_state.last_scoring_team;
      for (std::size_t idx = 0; idx < count; ++idx) {
        const CarState& car = current_state.cars[idx];
        const BallState& ball = current_state.ball;

        const float dx = car.position.x - ball.position.x;
        const float dy = car.position.y - ball.position.y;
        const float dz = car.position.z - ball.position.z;
        const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float step_proximity = (dist < 300.0F) ? 1.0F : 0.0F;

        const std::size_t global_idx = agent_begin + idx;
        const bool touch_edge = car.ball_touched && !agent_reward_states_[global_idx].prev_ball_touched;
        float& accumulated = episode_touch_ptr[global_idx];
        accumulated = std::max(accumulated, touch_edge ? 1.0F : 0.0F);
        if (touch_edge) {
          episode_touch_count_ptr[global_idx] += 1.0F;
        }
        ball_touch_ptr[global_idx] = step_proximity;

        const bool is_terminated = envs_[env_idx].terminated_scratch[idx] != 0;
        const bool is_truncated = envs_[env_idx].truncated_scratch[idx] != 0;
        const bool done = is_terminated || is_truncated;
        dones_ptr[global_idx] = done ? 1.0F : 0.0F;
        terminated_ptr[global_idx] = is_terminated ? 1.0F : 0.0F;
        truncated_ptr[global_idx] = is_truncated ? 1.0F : 0.0F;
        host_bootstrap_truncated_.data_ptr<float>()[global_idx] = (is_truncated && !is_terminated) ? 1.0F : 0.0F;

        const int label = done
            ? (goal_scored
                ? (car.team == scoring_team ? 0 : 1)
                : (episode_touch_count_ptr[global_idx] > 0.5F ? 2 : 3))
            : -1;
        if (done) {
          labels_ptr[global_idx] = label;
        }

        RewardBreakdown breakdown = reward_engine_.compute(
            current_state.tick, car, current_state,
            static_cast<int>(config_.env.team_size),
            agent_reward_states_[global_idx],
            env_reward_states_[env_idx],
            done, label);
        host_rewards_.data_ptr<float>()[global_idx] = breakdown.total;
        host_gameplay_rewards_.data_ptr<float>()[global_idx] = breakdown.gameplay;
        host_mechanic_rewards_.data_ptr<float>()[global_idx] = breakdown.mechanic;

        reset_needed = reset_needed || done;

        float goal_pos[3];
        compute_goal_position(current_state, config_.goal_mapping, goal_pos);
        const int pos_offset = static_cast<int>(global_idx) * 3;
        goal_pos_ptr[pos_offset + 0] = goal_pos[0];
        goal_pos_ptr[pos_offset + 1] = goal_pos[1];
        goal_pos_ptr[pos_offset + 2] = goal_pos[2];
      }

      // Team spirit blending: blend individual gameplay rewards with team average
      {
        const float team_spirit = config_.dense_rewards.team_spirit;
        if (team_spirit > 0.0F && count > 1) {
          float team0_sum = 0.0F, team1_sum = 0.0F;
          int team0_cnt = 0, team1_cnt = 0;
          for (std::size_t idx2 = 0; idx2 < count; ++idx2) {
            const std::size_t gidx = agent_begin + idx2;
            const float gp = host_gameplay_rewards_.data_ptr<float>()[gidx];
            if (current_state.cars[idx2].team == Team::Blue) {
              team0_sum += gp;
              team0_cnt++;
            } else {
              team1_sum += gp;
              team1_cnt++;
            }
          }
          const float team0_avg = team0_cnt > 0 ? team0_sum / static_cast<float>(team0_cnt) : 0.0F;
          const float team1_avg = team1_cnt > 0 ? team1_sum / static_cast<float>(team1_cnt) : 0.0F;
          for (std::size_t idx2 = 0; idx2 < count; ++idx2) {
            const std::size_t gidx = agent_begin + idx2;
            const float team_avg = (current_state.cars[idx2].team == Team::Blue) ? team0_avg : team1_avg;
            float* gp_ptr = host_gameplay_rewards_.data_ptr<float>() + gidx;
            const float old_gp = *gp_ptr;
            const float new_gp = (1.0F - team_spirit) * old_gp + team_spirit * team_avg;
            const float delta = new_gp - old_gp;
            *gp_ptr = new_gp;
            host_rewards_.data_ptr<float>()[gidx] += delta;
          }
        }
      }

      if (reset_needed) {
        obs_builder_->build_obs_batch(
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

  rebuild_next_buffers(timings);
  std::swap(current_buffers_, next_buffers_);
}

void BatchedRocketSimCollector::step(std::span<const ControllerState> actions, CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "step_controller");
  if (actions.size() != total_agents_) {
    throw std::invalid_argument("BatchedRocketSimCollector::step action span has incorrect size.");
  }

  const auto env_step_start = std::chrono::steady_clock::now();
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_begin = agent_offsets_[env_idx];
      const std::size_t agent_end = agent_offsets_[env_idx + 1];
      envs_[env_idx].engine->step_inplace(
          std::span<const ControllerState>(
              actions.data() + static_cast<std::ptrdiff_t>(agent_begin),
              agent_end - agent_begin));
    }
  });
  if (timings != nullptr) {
    timings->env_step_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - env_step_start).count();
  }
  finalize_step(timings);
}

void BatchedRocketSimCollector::step(std::span<const std::int64_t> action_indices, CollectorTimings* timings) {
  PULSAR_TRACE_SCOPE_CAT("collector", "step_discrete");
  if (action_indices.size() != total_agents_) {
    throw std::invalid_argument("BatchedRocketSimCollector::step action span has incorrect size.");
  }

  const auto env_step_start = std::chrono::steady_clock::now();
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_begin = agent_offsets_[env_idx];
      const std::size_t agent_end = agent_offsets_[env_idx + 1];
      const std::size_t count = agent_end - agent_begin;
      action_parser_->parse_actions_into(
          std::span<const std::int64_t>(
              action_indices.data() + static_cast<std::ptrdiff_t>(agent_begin),
              count),
          envs_[env_idx].action_scratch);
      envs_[env_idx].engine->step_inplace(envs_[env_idx].action_scratch);
    }
  });
  if (timings != nullptr) {
    timings->env_step_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - env_step_start).count();
  }
  finalize_step(timings);
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

const torch::Tensor& BatchedRocketSimCollector::host_env_touched() const {
  return host_env_touched_;
}

const torch::Tensor& BatchedRocketSimCollector::host_env_multi_touched() const {
  return host_env_multi_touched_;
}

}  // namespace pulsar

#endif
