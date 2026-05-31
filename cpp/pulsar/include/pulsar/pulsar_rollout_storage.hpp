// Pulsar — Phase 3: rollout storage.
//
// Time-major CPU tensors for one on-policy rollout. Mirrors sprint5's flat
// numpy buffers (sprint5:558-566):
//
//   obs_buf       [T, N_agents, 47]  float32
//   actions_buf   [T, N_agents,  8]  float32  (continuous)
//   logprobs_buf  [T, N_agents]      float32
//   rewards_buf   [T, N_agents]      float32
//   dones_buf     [T, N_agents]      float32
//   values_buf    [T, N_agents]      float32
//   future_goals  [T, N_agents,  6]  float32  (filled by hindsight sampler)
//   cells_buf     [T, N_agents,  2]  int32    (reachability grid cell ci/cj)
//
// N_agents = num_envs * 2  (blue + orange per env, interleaved: blue=even, orange=odd)
//
// Optional pinned memory (pin_host_memory=true) for non-blocking H→D transfers.
#pragma once

#ifdef PULSAR_HAS_TORCH
#include <cassert>
#include <cstdint>
#include <torch/torch.h>

namespace pulsar {

class PulsarRolloutStorage {
 public:
  static constexpr int kObsDim    = 47;
  static constexpr int kActionDim = 8;
  static constexpr int kGoalDim   = 6;

  PulsarRolloutStorage(int rollout_steps, int num_agents, bool pin_host_memory = false) {
    const auto opts = torch::TensorOptions(torch::kFloat32);
    const auto pin  = pin_host_memory
                          ? opts.pinned_memory(true)
                          : opts;

    obs         = torch::zeros({rollout_steps, num_agents, kObsDim},    pin);
    actions     = torch::zeros({rollout_steps, num_agents, kActionDim}, pin);
    logprobs    = torch::zeros({rollout_steps, num_agents},              pin);
    rewards     = torch::zeros({rollout_steps, num_agents},              pin);
    dones       = torch::zeros({rollout_steps, num_agents},              pin);
    values      = torch::zeros({rollout_steps, num_agents},              pin);
    future_goals = torch::zeros({rollout_steps, num_agents, kGoalDim},  pin);
    cells        = torch::zeros({rollout_steps, num_agents, 2},
                                torch::TensorOptions(torch::kInt32));

    steps_       = rollout_steps;
    num_agents_  = num_agents;
  }

  // ---- per-step write accessors -------------------------------------------
  // Called once per step from the training loop (after policy inference).

  void set_obs    (int t, const torch::Tensor& v) { obs.select(0, t).copy_(v);     }
  void set_actions(int t, const torch::Tensor& v) { actions.select(0, t).copy_(v); }
  void set_logprobs(int t, const torch::Tensor& v) { logprobs.select(0, t).copy_(v); }
  void set_rewards(int t, const torch::Tensor& v) { rewards.select(0, t).copy_(v); }
  void set_dones  (int t, const torch::Tensor& v) { dones.select(0, t).copy_(v);   }
  void set_values (int t, const torch::Tensor& v) { values.select(0, t).copy_(v);  }
  void set_cells  (int t, const torch::Tensor& v) { cells.select(0, t).copy_(v);   }

  // ---- dimensions ---------------------------------------------------------
  int steps()      const { return steps_;      }
  int num_agents() const { return num_agents_; }

  // ---- public tensors (the training loop accesses them directly) ----------
  torch::Tensor obs;          // [T, N, 47]
  torch::Tensor actions;      // [T, N, 8]
  torch::Tensor logprobs;     // [T, N]
  torch::Tensor rewards;      // [T, N]
  torch::Tensor dones;        // [T, N]
  torch::Tensor values;       // [T, N]
  torch::Tensor future_goals; // [T, N, 6]  — filled post-rollout
  torch::Tensor cells;        // [T, N, 2]  — reachability grid cell (int32)

 private:
  int steps_;
  int num_agents_;
};

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
