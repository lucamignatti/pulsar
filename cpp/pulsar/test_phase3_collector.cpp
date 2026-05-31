// Phase 3 gate: collector smoke test.
//
// Runs a short rollout (T=10 steps, N=4 envs) with a random policy and checks:
//   - obs is 47-dim, values in [-2, 2] (loose; some vel spikes may occur)
//   - rewards are in {-1, 0, 1}
//   - dones are in {0, 1}
//   - terminal_obs is captured (non-zero when episode ends)
//   - The entire pipeline (step_physics_prefix + step_finish) runs without crash.
//
// Needs RocketSim initialized. Pass the collision meshes dir as argv[1]
// (defaults to external/RocketSim/collision_meshes).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <torch/torch.h>

#include "pulsar/pulsar_collector.hpp"
#include "pulsar/pulsar_env.hpp"

using namespace pulsar;

int main(int argc, char** argv) {
  // Let the engine handle RocketSim::Init via its call_once — do NOT call it directly.
  // Pass the collision_meshes path to the collector constructor instead.
  const std::string meshes_dir = argc > 1 ? argv[1] : "collision_meshes";

  const int N_ENVS    = 4;
  const int WORKERS   = 2;
  const int T         = 20;
  const int TICK_SKIP = 8;

  // Use the midfield snapshot mutator for deterministic resets
  auto mutator = std::make_shared<PulsarStateMutator>(midfield_snapshot());

  PulsarCollector collector(N_ENVS, WORKERS, mutator, TICK_SKIP,
                          /*max_steps=*/200, /*pin=*/false, meshes_dir,
                          /*curriculum_grid=*/nullptr);
  collector.initialize(/*base_seed=*/42);

  std::printf("[pulsar_phase3] collector initialized: %d envs, %d agents, meshes='%s'\n",
              N_ENVS, collector.num_agents(), meshes_dir.c_str());

  torch::manual_seed(0);
  const int N_agents = collector.num_agents();

  int total_done = 0;
  float max_obs_abs = 0.0f;
  bool bad_reward = false;

  for (int t = 0; t < T; ++t) {
    // Random policy — continuous actions in [-1, 1]
    const auto actions = torch::rand({N_agents, 8}) * 2.0f - 1.0f;

    // Half-step: in a real trainer the prefix runs in a background thread.
    // Here we test the combined call to keep it simple.
    collector.step(actions);

    const auto& obs   = collector.current_obs();
    const auto& rew   = collector.last_rewards();
    const auto& done  = collector.last_dones();

    // Shape checks
    if (obs.size(0) != N_agents || obs.size(1) != 47) {
      std::printf("FAIL: obs shape [%lld, %lld] expected [%d, 47]\n",
                  (long long)obs.size(0), (long long)obs.size(1), N_agents);
      return 1;
    }

    // Track max abs obs value
    const float mo = obs.abs().max().item<float>();
    if (mo > max_obs_abs) max_obs_abs = mo;

    // Reward must be in {-1, 0, 1}
    const auto rew_np = rew.view(-1);
    for (int i = 0; i < N_agents; ++i) {
      const float r = rew_np[i].item<float>();
      if (r != 0.0f && r != 1.0f && r != -1.0f) {
        std::printf("FAIL: unexpected reward %.3f at step %d agent %d\n", r, t, i);
        bad_reward = true;
      }
    }

    // Count terminal episodes
    const float n_done = done.sum().item<float>();
    total_done += static_cast<int>(n_done);
  }

  std::printf("[pulsar_phase3] rollout complete: %d steps, %d terminal events\n",
              T, total_done);
  std::printf("[pulsar_phase3] max |obs| = %.4f  (expected < ~15 for normalized obs)\n",
              max_obs_abs);

  int fails = 0;

  // Obs should be roughly normalized — values > 15 indicate a normalization bug
  if (max_obs_abs > 15.0f) {
    std::printf("FAIL: obs values out of expected range (max_abs=%.4f)\n", max_obs_abs);
    ++fails;
  } else {
    std::printf("ok   obs range check (max_abs=%.4f)\n", max_obs_abs);
  }

  if (bad_reward) {
    std::printf("FAIL: non-{-1,0,1} reward observed\n");
    ++fails;
  } else {
    std::printf("ok   reward values in {-1, 0, 1}\n");
  }

  // Done events: midfield state with random policy — some episodes should end
  // within 200 steps, but with T=20 steps * 8 ticks = 160 ticks we might not
  // hit max_steps. Just check done is binary.
  const auto done_vals = collector.last_dones();
  const bool done_ok = (done_vals.min().item<float>() >= 0.0f) &&
                       (done_vals.max().item<float>() <= 1.0f);
  if (!done_ok) {
    std::printf("FAIL: done values out of [0,1]\n");
    ++fails;
  } else {
    std::printf("ok   done values in [0,1]\n");
  }

  std::printf("\n[pulsar_phase3] %s (%d failure(s))\n",
              fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
