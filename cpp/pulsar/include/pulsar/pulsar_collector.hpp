// Pulsar — Phase 3: parallel environment collector.
//
// Lean reimplementation of the batched_rocketsim_collector pattern for the
// continuous-action, terminal-reward, 1v1 self-play setting.
//
// Key design decisions vs the legacy BatchedRocketSimCollector:
//   - Actions are float[N_agents * 8] (continuous), not int64 action indices.
//   - No action masks, no discrete parser.
//   - Rewards are terminal-only ±1 (PulsarDoneCondition + reward_from_outcome).
//   - No multi-head goals — a single goal dim=6 vector is managed externally.
//   - No x-mirror augmentation (handled in PulsarContinuousActionParser).
//   - Envs are always 1v1: 2 agents per env, `N_agents = num_envs * 2`.
//
// Half-stepping (same latency-hiding trick as the legacy collector):
//   1. Trainer calls step_physics_prefix(prefix_ticks) on a background thread
//      while computing the policy forward pass.
//   2. Trainer calls step_finish(actions, remaining_ticks) on the main thread.
//      This applies new actions, runs remaining physics, checks done, resets,
//      and builds the next step's observations.
//
// Outputs (pinned CPU tensors, zero-copy to GPU via non_blocking=true):
//   current_obs()       → [N_agents, 47]
//   last_rewards()      → [N_agents]
//   last_dones()        → [N_agents]
//   last_terminal_obs() → [N_agents, 47]   (obs at episode end, pre-reset)
//   last_episode_starts() → [N_agents]     (1 if this is a fresh episode)
//   last_cells()        → [N_agents, 2]    (reachability grid ci/cj, int32)
#pragma once

#ifdef PULSAR_HAS_TORCH
#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include <torch/torch.h>

#include "pulsar/core/parallel_executor.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/pulsar_env.hpp"

namespace pulsar {

class PulsarReachabilityGrid;  // forward decl (avoid circular include)

// Per-environment runtime state.
struct PulsarEnvRuntime {
  std::shared_ptr<RocketSimTransitionEngine> engine;
  std::shared_ptr<PulsarStateMutator> per_env_mutator;  // owned snapshot, writable
  std::vector<ControllerState> action_scratch;  // [2], current applied actions
  int episode_ticks = 0;                        // tick counter for truncation
  ::std::mt19937 curriculum_rng{0};              // per-env RNG for grid sampling
  // Reachability grid cell for each agent (ci, cj).
  int32_t cell_ci[2] = {0, 0};
  int32_t cell_cj[2] = {0, 0};
};

class PulsarCollector {
 public:
  // `num_envs`: number of parallel 1v1 arenas.
  // `num_workers`: thread-pool size for parallel stepping.
  // `mutator`: sampled on every episode reset (e.g. PulsarStateMutator or curriculum mutator).
  // `tick_skip`: physics ticks per logical step (default 8, matching sprint5).
  // `max_steps`: max logical steps before truncation (default 200, matching sprint5).
  // `collision_meshes_path`: passed to RocketSim::Init (must call once before first env).
  // `pin_host_memory`: pin output tensors for faster H→D DMA.
  // `mutator`: used for the first episode reset of each env. Pass nullptr to use
  // the default kickoff. Pass a `PulsarStateMutator` with an initial snapshot; the
  // collector will update it per-reset when `curriculum_grid` is set.
  // `curriculum_grid`: if non-null, the collector samples start states from the
  // grid on every episode reset and updates the cell indices.
  PulsarCollector(int num_envs, int num_workers,
               StateMutatorPtr mutator,
               int tick_skip = 8,
               int max_steps = 200,
               bool pin_host_memory = false,
               std::string collision_meshes_path = "collision_meshes",
               PulsarReachabilityGrid* curriculum_grid = nullptr);

  // Initialize all environments and build the initial observation tensors.
  // Must be called once after construction (and after RocketSim::Init).
  void initialize(std::uint64_t base_seed = 0);

  // Switch all envs to standard RocketSim random kickoff resets.
  // Called once when the curriculum grid is fully mastered.
  // After this call, curriculum_grid_ is cleared and the per-env mutators are
  // replaced with kickoff-mode engines — no snapshot injection on reset.
  void switch_to_kickoff(std::uint64_t base_seed = 0);

  // ---- Half-stepping interface -------------------------------------------

  // Apply current action_scratch + run prefix_ticks physics ticks.
  // Called on a background thread while the policy forward pass runs.
  void step_physics_prefix(int prefix_ticks);

  // Apply new continuous actions [N_agents, 8] + run remaining_ticks ticks,
  // check done/reset, build next obs. Returns immediately after updating
  // all output tensors.
  // `remaining_ticks = tick_skip - prefix_ticks`.
  void step_finish(const torch::Tensor& actions_f, int remaining_ticks);

  // Non-half-stepping convenience (step_physics_prefix(0) + step_finish(a, tick_skip)).
  void step(const torch::Tensor& actions_f);

  // ---- Output accessors (valid after initialize() and each step()) -------
  const torch::Tensor& current_obs()          const { return host_obs_; }
  const torch::Tensor& last_rewards()         const { return host_rewards_; }
  const torch::Tensor& last_dones()           const { return host_dones_; }
  const torch::Tensor& last_terminal_obs()    const { return host_terminal_obs_; }
  const torch::Tensor& last_episode_starts()  const { return host_episode_starts_; }
  const torch::Tensor& last_cells()           const { return host_cells_; }

  int num_envs()   const { return num_envs_; }
  int num_agents() const { return num_envs_ * 2; }

 private:
  void rebuild_obs_for_env(std::size_t env_idx);
  float reward_from_outcome(const EnvState& state, int agent_id) const;

  int num_envs_;
  int tick_skip_;
  int max_episode_ticks_;
  std::string collision_meshes_path_;
  StateMutatorPtr initial_mutator_;    // used for first-episode reset
  PulsarReachabilityGrid* curriculum_grid_ = nullptr;
  PulsarObsBuilder obs_builder_;
  PulsarDoneCondition done_condition_;

  std::vector<PulsarEnvRuntime> envs_;
  ParallelExecutor executor_;

  // Pinned host tensors (optionally pinned)
  torch::Tensor host_obs_;            // [N_agents, 47]  current obs
  torch::Tensor host_rewards_;        // [N_agents]
  torch::Tensor host_dones_;          // [N_agents]
  torch::Tensor host_terminal_obs_;   // [N_agents, 47]  obs at episode end
  torch::Tensor host_episode_starts_; // [N_agents]  1 = new episode this step
  torch::Tensor host_cells_;          // [N_agents, 2] int32
};

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
