// Pulsar — Phase 3: PulsarCollector implementation.
// See pulsar_collector.hpp for the design spec.
#include "pulsar/pulsar_collector.hpp"
#include "pulsar/pulsar_curriculum.hpp"  // PulsarReachabilityGrid

#ifdef PULSAR_HAS_TORCH
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "pulsar/config/config.hpp"

namespace pulsar {

namespace {

EnvConfig make_pulsar_env_config(std::uint64_t seed, int tick_skip, int max_episode_ticks,
                               const std::string& meshes_path, bool use_kickoff = false) {
  EnvConfig cfg;
  cfg.team_size          = 1;
  cfg.tick_skip          = tick_skip;
  cfg.max_episode_ticks  = max_episode_ticks;
  // "random" in mode triggers the engine's mutator path (applies snapshot to arena).
  // Anything else triggers arena_->ResetToRandomKickoff() — used post-curriculum.
  cfg.mode               = use_kickoff ? "kickoff" : "random_pulsar";
  cfg.seed               = seed;
  if (!meshes_path.empty()) cfg.collision_meshes_path = meshes_path;
  return cfg;
}

}  // namespace

PulsarCollector::PulsarCollector(int num_envs, int num_workers,
                             StateMutatorPtr mutator,
                             int tick_skip, int max_steps,
                             bool pin_host_memory,
                             std::string collision_meshes_path,
                             PulsarReachabilityGrid* curriculum_grid)
    : num_envs_(num_envs),
      tick_skip_(tick_skip),
      max_episode_ticks_(max_steps * tick_skip),
      collision_meshes_path_(std::move(collision_meshes_path)),
      initial_mutator_(std::move(mutator)),
      curriculum_grid_(curriculum_grid),
      done_condition_(max_steps * tick_skip),
      executor_(static_cast<std::size_t>(num_workers)) {
  const int N = num_envs * 2;
  const auto opts = torch::TensorOptions(torch::kFloat32);
  const auto pin  = pin_host_memory ? opts.pinned_memory(true) : opts;

  host_obs_            = torch::zeros({N, PulsarObsBuilder::kObsDim}, pin);
  host_rewards_        = torch::zeros({N}, pin);
  host_dones_          = torch::zeros({N}, pin);
  host_terminal_obs_   = torch::zeros({N, PulsarObsBuilder::kObsDim}, pin);
  host_episode_starts_ = torch::zeros({N}, pin);
  host_cells_          = torch::zeros({N, 2}, torch::TensorOptions(torch::kInt32));
}

void PulsarCollector::initialize(std::uint64_t base_seed) {
  envs_.clear();
  envs_.reserve(static_cast<std::size_t>(num_envs_));

  for (int ei = 0; ei < num_envs_; ++ei) {
    PulsarEnvRuntime rt;
    rt.curriculum_rng = std::mt19937(base_seed + static_cast<uint64_t>(ei) * 1000 + 1);

    // Each env gets its own writable PulsarStateMutator — the collector updates it
    // before each reset. Starting snapshot: curriculum grid or midfield fallback.
    PulsarEnvSnapshot initial_snap = midfield_snapshot();
    int initial_ci = PulsarReachabilityGrid::kXBins / 2;
    int initial_cj = PulsarReachabilityGrid::kYBins / 2;
    if (curriculum_grid_) {
      auto [snap, ci, cj] = curriculum_grid_->sample_start_state(rt.curriculum_rng);
      initial_snap = snap;
      initial_ci = ci;
      initial_cj = cj;
    }
    rt.cell_ci[0] = rt.cell_ci[1] = initial_ci;
    rt.cell_cj[0] = rt.cell_cj[1] = initial_cj;
    rt.per_env_mutator = std::make_shared<PulsarStateMutator>(initial_snap);

    EnvConfig cfg = make_pulsar_env_config(
        base_seed + static_cast<std::uint64_t>(ei),
        tick_skip_, max_episode_ticks_, collision_meshes_path_);
    rt.engine = std::make_shared<RocketSimTransitionEngine>(cfg, rt.per_env_mutator);
    rt.action_scratch.resize(2);
    rt.episode_ticks = 0;

    rt.engine->reset(base_seed + static_cast<std::uint64_t>(ei));
    envs_.push_back(std::move(rt));
  }

  for (int ei = 0; ei < num_envs_; ++ei) {
    rebuild_obs_for_env(static_cast<std::size_t>(ei));
  }
  host_episode_starts_.fill_(1.0f);
  host_dones_.fill_(0.0f);
  host_rewards_.fill_(0.0f);
}

void PulsarCollector::switch_to_kickoff(std::uint64_t base_seed) {
  curriculum_grid_ = nullptr;  // stop curriculum sampling

  for (int ei = 0; ei < num_envs_; ++ei) {
    EnvConfig cfg = make_pulsar_env_config(
        base_seed + static_cast<std::uint64_t>(ei),
        tick_skip_, max_episode_ticks_, collision_meshes_path_,
        /*use_kickoff=*/true);
    // No mutator: engine takes the kickoff path (arena_->ResetToRandomKickoff)
    envs_[ei].per_env_mutator = nullptr;
    envs_[ei].engine = std::make_shared<RocketSimTransitionEngine>(cfg, nullptr);
    envs_[ei].episode_ticks = 0;
    envs_[ei].cell_ci[0] = envs_[ei].cell_ci[1] = 0;
    envs_[ei].cell_cj[0] = envs_[ei].cell_cj[1] = 0;
    envs_[ei].engine->reset(base_seed + static_cast<std::uint64_t>(ei));
    rebuild_obs_for_env(static_cast<std::size_t>(ei));
  }
  host_episode_starts_.fill_(1.0f);
  host_dones_.fill_(0.0f);
  host_rewards_.fill_(0.0f);
}

void PulsarCollector::rebuild_obs_for_env(std::size_t env_idx) {
  const EnvState& state = envs_[env_idx].engine->state();
  const std::size_t base_agent = env_idx * 2;
  float* obs_ptr = host_obs_.data_ptr<float>() +
                   base_agent * PulsarObsBuilder::kObsDim;
  std::span<float> obs_span{obs_ptr, static_cast<std::size_t>(2 * PulsarObsBuilder::kObsDim)};
  obs_builder_.build_obs_batch(state, obs_span);
}

float PulsarCollector::reward_from_outcome(const EnvState& state, int agent_id) const {
  if (state.goal_scored) {
    const bool blue_scored = (state.last_scoring_team == Team::Blue);
    const bool is_blue     = (agent_id == 0);
    return (blue_scored == is_blue) ? 1.0f : -1.0f;
  }
  return 0.0f;
}

void PulsarCollector::step_physics_prefix(int prefix_ticks) {
  if (prefix_ticks <= 0) return;
  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t ei = begin; ei < end; ++ei) {
      envs_[ei].engine->apply_controls(envs_[ei].action_scratch);
      envs_[ei].engine->step_physics(prefix_ticks);
    }
  });
}

void PulsarCollector::step_finish(const torch::Tensor& actions_f, int remaining_ticks) {
  assert(actions_f.size(0) == num_agents() && actions_f.size(1) == 8);
  assert(actions_f.device().is_cpu());

  const float* actions_ptr = actions_f.contiguous().data_ptr<float>();
  float*    reward_ptr   = host_rewards_.data_ptr<float>();
  float*    done_ptr     = host_dones_.data_ptr<float>();
  float*    term_obs_ptr = host_terminal_obs_.data_ptr<float>();
  float*    ep_start_ptr = host_episode_starts_.data_ptr<float>();
  int32_t*  cells_ptr    = host_cells_.data_ptr<int32_t>();

  executor_.parallel_for(envs_.size(), [&](std::size_t begin, std::size_t end) {
    for (std::size_t ei = begin; ei < end; ++ei) {
      const std::size_t base_agent = ei * 2;

      for (int ai = 0; ai < 2; ++ai) {
        const float* av = actions_ptr + (base_agent + ai) * 8;
        envs_[ei].action_scratch[ai] = PulsarContinuousActionParser::parse(av, ai);
      }

      auto& engine = *envs_[ei].engine;
      if (remaining_ticks > 0) {
        engine.apply_controls(envs_[ei].action_scratch);
        engine.step_physics(remaining_ticks);
      } else {
        engine.step_inplace(envs_[ei].action_scratch);
      }
      envs_[ei].episode_ticks += tick_skip_;

      const EnvState& state = engine.state();

      std::uint8_t term_s[2], trunc_s[2];
      done_condition_.is_done_into(state, envs_[ei].episode_ticks,
                                   std::span<uint8_t>(term_s, 2),
                                   std::span<uint8_t>(trunc_s, 2));
      const bool done = term_s[0] || trunc_s[0];

      for (int ai = 0; ai < 2; ++ai) {
        reward_ptr[base_agent + ai] = done ? reward_from_outcome(state, ai) : 0.0f;
        done_ptr  [base_agent + ai] = done ? 1.0f : 0.0f;
      }

      if (done) {
        // Capture terminal obs
        std::span<float> term_span{
            term_obs_ptr + base_agent * PulsarObsBuilder::kObsDim,
            static_cast<std::size_t>(2 * PulsarObsBuilder::kObsDim)};
        obs_builder_.build_obs_batch(state, term_span);

        // Sample next start state (curriculum) or let engine do kickoff (post-curriculum)
        if (curriculum_grid_) {
          // Thread-safe: each env has its own curriculum_rng and per_env_mutator.
          auto [snap, ci, cj] = curriculum_grid_->sample_start_state(
              envs_[ei].curriculum_rng);
          envs_[ei].per_env_mutator->set_snapshot(snap);
          envs_[ei].cell_ci[0] = envs_[ei].cell_ci[1] = ci;
          envs_[ei].cell_cj[0] = envs_[ei].cell_cj[1] = cj;
        }
        // If curriculum_grid_ == nullptr, engine is in kickoff mode and handles
        // arena_->ResetToRandomKickoff() internally — no mutator action needed.

        const std::uint64_t new_seed = static_cast<std::uint64_t>(
            envs_[ei].episode_ticks * 1000 + static_cast<int>(ei));
        envs_[ei].episode_ticks = 0;
        engine.reset(new_seed);
      }

      rebuild_obs_for_env(ei);

      ep_start_ptr[base_agent]     = done ? 1.0f : 0.0f;
      ep_start_ptr[base_agent + 1] = done ? 1.0f : 0.0f;

      for (int ai = 0; ai < 2; ++ai) {
        cells_ptr[(base_agent + ai) * 2 + 0] = envs_[ei].cell_ci[ai];
        cells_ptr[(base_agent + ai) * 2 + 1] = envs_[ei].cell_cj[ai];
      }
    }
  });
}

void PulsarCollector::step(const torch::Tensor& actions_f) {
  step_finish(actions_f, tick_skip_);
}

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
