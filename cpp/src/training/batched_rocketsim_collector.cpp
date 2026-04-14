#include "pulsar/training/batched_rocketsim_collector.hpp"

#ifdef PULSAR_HAS_TORCH

#include <chrono>
#include <stdexcept>

#include "pulsar/rl/action_table.hpp"

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

}  // namespace

BatchedRocketSimCollector::BatchedRocketSimCollector(
    ExperimentConfig config,
    ObsBuilderPtr obs_builder,
    ActionParserPtr action_parser,
    DoneConditionPtr done_condition,
    bool pin_host_memory)
    : BatchedRocketSimCollector(
          config,
          make_default_engines(config),
          std::move(obs_builder),
          std::move(action_parser),
          std::move(done_condition),
          pin_host_memory) {}

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
      executor_(static_cast<std::size_t>(config_.ppo.collection_workers)) {
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
    envs_.push_back(EnvRuntime{
        .engine = std::move(engines[env_idx]),
        .assignment = {},
        .reset_seed = config_.env.seed + static_cast<std::uint64_t>(env_idx),
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

  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  if (pin_host_memory) {
    f32 = f32.pinned_memory(true);
    u8 = u8.pinned_memory(true);
    i64 = i64.pinned_memory(true);
  }

  host_obs_ = torch::empty({static_cast<long>(total_agents_), obs_dim_}, f32);
  host_action_masks_ = torch::empty({static_cast<long>(total_agents_), action_dim_}, u8);
  host_learner_active_ = torch::ones({static_cast<long>(total_agents_)}, f32);
  host_snapshot_ids_ = torch::full({static_cast<long>(total_agents_)}, -1, i64);
  host_episode_starts_ = torch::ones({static_cast<long>(total_agents_)}, f32);
  host_dones_ = torch::zeros({static_cast<long>(total_agents_)}, f32);
  host_terminal_next_goal_labels_ = torch::full({static_cast<long>(total_agents_)}, 2, i64);
  host_post_step_obs_ = torch::empty({static_cast<long>(total_agents_), obs_dim_}, f32);

  for (std::size_t env_idx = 0; env_idx < envs_.size(); ++env_idx) {
    assign_env(env_idx, envs_[env_idx].reset_seed);
  }
}

void BatchedRocketSimCollector::set_self_play_assignment_fn(AssignmentFn assignment_fn) {
  assignment_fn_ = std::move(assignment_fn);
  for (std::size_t env_idx = 0; env_idx < envs_.size(); ++env_idx) {
    assign_env(env_idx, envs_[env_idx].reset_seed);
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

torch::Tensor BatchedRocketSimCollector::collect_observations(CollectorTimings* timings) {
  const auto start = std::chrono::steady_clock::now();
  float* dst = host_obs_.data_ptr<float>();
  const std::size_t stride = static_cast<std::size_t>(obs_dim_);
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_offset = agent_offsets_[env_idx];
      const std::size_t count = envs_[env_idx].engine->num_agents();
      obs_builder_->build_obs_batch(
          envs_[env_idx].engine->state(),
          std::span<float>(
              dst + static_cast<std::ptrdiff_t>(agent_offset * stride),
              count * stride));
    }
  });
  if (timings != nullptr) {
    timings->obs_build_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
  return host_obs_;
}

void BatchedRocketSimCollector::collect_action_masks(CollectorTimings* timings) {
  const auto start = std::chrono::steady_clock::now();
  std::uint8_t* masks_ptr = host_action_masks_.data_ptr<std::uint8_t>();
  float* learner_ptr = host_learner_active_.data_ptr<float>();
  std::int64_t* snapshot_ptr = host_snapshot_ids_.data_ptr<std::int64_t>();
  const std::size_t stride = static_cast<std::size_t>(action_dim_);

  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const EnvState& state = envs_[env_idx].engine->state();
      const std::size_t agent_offset = agent_offsets_[env_idx];
      action_parser_->build_action_mask_batch(
          state,
          std::span<std::uint8_t>(
              masks_ptr + static_cast<std::ptrdiff_t>(agent_offset * stride),
              state.cars.size() * stride));

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
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
}

void BatchedRocketSimCollector::step(
    std::span<const ControllerState> actions,
    bool collect_post_step_obs,
    CollectorTimings* timings) {
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

  const auto done_reset_start = std::chrono::steady_clock::now();
  float* dones_ptr = host_dones_.data_ptr<float>();
  std::int64_t* labels_ptr = host_terminal_next_goal_labels_.data_ptr<std::int64_t>();
  float* post_step_obs_ptr = host_post_step_obs_.data_ptr<float>();
  host_dones_.zero_();
  host_terminal_next_goal_labels_.fill_(2);

  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t env_idx = begin; env_idx < end; ++env_idx) {
      const std::size_t agent_begin = agent_offsets_[env_idx];
      const std::size_t agent_end = agent_offsets_[env_idx + 1];
      const std::size_t count = agent_end - agent_begin;
      const EnvState& current_state = envs_[env_idx].engine->state();

      std::vector<std::uint8_t> terminated(count, 0);
      std::vector<std::uint8_t> truncated(count, 0);
      done_condition_->is_done_into(current_state, current_state.tick, terminated, truncated);

      if (collect_post_step_obs) {
        obs_builder_->build_obs_batch(
            current_state,
            std::span<float>(
                post_step_obs_ptr + static_cast<std::ptrdiff_t>(agent_begin * static_cast<std::size_t>(obs_dim_)),
                count * static_cast<std::size_t>(obs_dim_)));
      }

      bool reset_needed = false;
      Team scoring_team = Team::Blue;
      bool goal_scored = current_state.goal_scored;
      if (goal_scored) {
        scoring_team = current_state.blue_score > current_state.orange_score ? Team::Blue : Team::Orange;
      }
      for (std::size_t idx = 0; idx < count; ++idx) {
        const bool done = terminated[idx] != 0 || truncated[idx] != 0;
        dones_ptr[agent_begin + idx] = done ? 1.0F : 0.0F;
        if (done) {
          labels_ptr[agent_begin + idx] =
              goal_scored ? (current_state.cars[idx].team == scoring_team ? 0 : 1) : 2;
        }
        reset_needed = reset_needed || done;
      }

      if (reset_needed) {
        envs_[env_idx].reset_seed += static_cast<std::uint64_t>(envs_.size());
        envs_[env_idx].engine->reset(envs_[env_idx].reset_seed);
        assign_env(env_idx, envs_[env_idx].reset_seed);
      }
    }
  });

  host_episode_starts_.copy_(host_dones_);
  if (timings != nullptr) {
    timings->done_reset_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - done_reset_start).count();
  }
}

const torch::Tensor& BatchedRocketSimCollector::host_action_masks() const {
  return host_action_masks_;
}

const torch::Tensor& BatchedRocketSimCollector::host_learner_active() const {
  return host_learner_active_;
}

const torch::Tensor& BatchedRocketSimCollector::host_snapshot_ids() const {
  return host_snapshot_ids_;
}

const torch::Tensor& BatchedRocketSimCollector::host_episode_starts() const {
  return host_episode_starts_;
}

const torch::Tensor& BatchedRocketSimCollector::host_dones() const {
  return host_dones_;
}

const torch::Tensor& BatchedRocketSimCollector::host_terminal_next_goal_labels() const {
  return host_terminal_next_goal_labels_;
}

const torch::Tensor& BatchedRocketSimCollector::host_post_step_obs() const {
  return host_post_step_obs_;
}

void BatchedRocketSimCollector::assign_env(std::size_t env_idx, std::uint64_t seed) {
  if (assignment_fn_) {
    envs_[env_idx].assignment = assignment_fn_(env_idx, seed);
  } else {
    envs_[env_idx].assignment = {};
  }
}

}  // namespace pulsar

#endif
