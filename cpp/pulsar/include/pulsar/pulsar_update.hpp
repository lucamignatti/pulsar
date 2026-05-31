// Pulsar — Phase 4: update loop.
//
// Implements the full PPO + quasimetric-advantage + InfoNCE update from
// sprint5:649-758 in C++. No legacy pulsar_torch / actor.hpp dependency;
// all math is implemented directly against the Pulsar model types.
//
// Key design points:
//   - GAE matches sprint5:660-678 exactly (gamma=0.99, lambda=0.95).
//   - Quasimetric advantage = normalize(q - mean(q)) sprint5:686-691.
//   - PPO clipped surrogate: max(-adv*ratio, -adv*clamp(ratio,0.8,1.2))
//     (note: loss sign is positive; optimizer minimises) sprint5:725.
//   - Gaussian log_prob / entropy computed directly (no torch::Normal).
//   - Offense (even agent_idx) weight=1.0; defense (odd) weight=0.85.
//   - Advantage sign flip for Orange/defender before per-minibatch norm.
//   - Three Adam optimisers (lr=3e-4), grad clip 0.5 per network.
#pragma once

#ifdef PULSAR_HAS_TORCH
#include <cmath>
#include <torch/torch.h>
#include "pulsar/pulsar_models.hpp"
#include "pulsar/pulsar_rollout_storage.hpp"

namespace pulsar {

// ============================================================================
// Standalone math utilities (used by the parity test and the training loop)
// ============================================================================

// GAE computation — exactly matches sprint5:660-678.
// rewards/values/dones : [T, N]
// next_values/next_done : [N]
// Returns advantages [T, N]; call returns = advantages + values.
torch::Tensor compute_gae_pulsar(
    const torch::Tensor& rewards,
    const torch::Tensor& values,
    const torch::Tensor& dones,
    const torch::Tensor& next_values,
    const torch::Tensor& next_done,
    float gamma = 0.99f,
    float gae_lambda = 0.95f);

// Quasimetric advantage — sprint5:686-691.
// q [B]: raw critic scores from PulsarQuasimetricCritic::forward().
// Returns normalised advantage [B].
torch::Tensor compute_quasimetric_advantage(const torch::Tensor& q);

// Gaussian log-prob (sum over action dims).  [B, D] → [B].
// sprint5: pr.log_prob(act).sum(-1)
torch::Tensor gaussian_log_prob(
    const torch::Tensor& actions,
    const torch::Tensor& mean,
    const torch::Tensor& std);

// Gaussian entropy (sum over action dims).  [B, D] → [B].
// sprint5: pr.entropy().sum(-1)
torch::Tensor gaussian_entropy(const torch::Tensor& std);

// Hindsight goal sampler — sprint5:649-658.
// obs_buf [T, N, 47], dones_buf [T, N] → future_goals [T, N, 6].
// goal = obs_buf[t + offset, e, 19:25] where offset ~ Geometric(p=0.2).
// Uses the provided RNG generator (pass a CPU generator with a fixed seed for parity).
torch::Tensor sample_hindsight_goals(
    const torch::Tensor& obs_buf,
    const torch::Tensor& dones_buf,
    float geom_p = 0.2f);

// ============================================================================
// PulsarUpdateResult — per-update diagnostics
// ============================================================================
struct PulsarUpdateResult {
  float pg_loss;
  float value_loss;
  float entropy;
  float infonce_loss;
  float total_loss;
  float approx_kl;
};

// ============================================================================
// PulsarUpdateLoop — holds 3 optimisers, runs the full PPO+InfoNCE update
// ============================================================================
class PulsarUpdateLoop {
 public:
  // Fixed goal vectors matching sprint5:569-570.
  static torch::Tensor goal_score_tensor(torch::Device dev);
  static torch::Tensor goal_defend_tensor(torch::Device dev);

  // Build goal tensor [N_agents, 6]: even=score, odd=defend.
  static torch::Tensor build_agent_goals(int num_agents, torch::Device dev);

  PulsarUpdateLoop(PulsarActor actor, PulsarValue value, PulsarQuasimetricCritic critic,
                float lr = 3e-4f);

  // Run one full rollout update (sprint5:693-758).
  // rollout : pre-filled PulsarRolloutStorage (obs/actions/logprobs/rewards/dones/values)
  // next_obs [N, 47], next_done [N] : bootstrap tensors
  // num_epochs, minibatch_size : PPO hyper-parameters
  // device : target device for GPU computation (MPS / CUDA / CPU)
  PulsarUpdateResult update(
      PulsarRolloutStorage& rollout,
      const torch::Tensor& next_obs,
      const torch::Tensor& next_done,
      int num_epochs = 4,
      int minibatch_size = 1024,
      torch::Device device = torch::kCPU);

  // Zero all three optimisers.
  void zero_grad();

 private:
  PulsarActor actor_;
  PulsarValue value_;
  PulsarQuasimetricCritic critic_;
  torch::optim::Adam actor_opt_;
  torch::optim::Adam value_opt_;
  torch::optim::Adam critic_opt_;
};

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
