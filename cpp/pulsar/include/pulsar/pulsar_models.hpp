// Pulsar — Phase 2: faithful model port.
//
// Three tiny MLPs exactly mirroring the sprint5 Python prototype:
//
//   PulsarActor          — 128×128 tanh, Gaussian head, goal residual
//                       (sprint5:457-477, ActorModel)
//   PulsarValue          — 128×128×1 tanh over [obs|goal]
//                       (sprint5:479-488, ValueModel)
//   PulsarQuasimetricCritic — phi([obs|act])→repr32, psi(goal)→repr32,
//                          ASYMMETRIC softplus distance  (sprint5:490-508)
//
// All layer_init from sprint5 (orthogonal weight, std=sqrt(2) for hidden,
// std=0.01 for policy head, std=1.0 for value output).
//
// Uses LibTorch (PULSAR_HAS_TORCH). No custom CUDA/Metal kernels — MPS via
// PyTorch's built-in backend.
#pragma once

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>

namespace pulsar {

// Dimensions (matching sprint5 constants)
inline constexpr int kObsDim    = 47;
inline constexpr int kActionDim = 8;
inline constexpr int kGoalDim   = 6;
inline constexpr int kHiddenDim = 128;
inline constexpr int kReprDim   = 32;

// ============================================================================
// PulsarActor  —  sprint5 ActorModel
// ============================================================================
// Forward:  (obs[B,47], goal[B,6]) → (mean[B,8], std[B,8], features[B,128])
//
// Named params (match Python state_dict exactly for weight loading):
//   base.0.*  base.2.*  actor_mean.*  actor_logstd  goal_proj.*
//
// Orange action mirroring is done OUTSIDE the model (in PulsarContinuousActionParser).

struct PulsarActorImpl : torch::nn::Module {
  torch::nn::Sequential base{nullptr};
  torch::nn::Linear     actor_mean{nullptr};
  torch::nn::Linear     goal_proj{nullptr};
  torch::Tensor         actor_logstd;

  PulsarActorImpl();

  // Returns (mean, std, features). goal may be empty ({0}) to skip goal conditioning.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor obs, torch::Tensor goal);
};
TORCH_MODULE(PulsarActor);

// ============================================================================
// PulsarValue  —  sprint5 ValueModel
// ============================================================================
// Forward:  (obs[B,47], goal[B,6]) → value[B,1]
//
// Named params:  base.0.*  base.2.*  base.4.*

struct PulsarValueImpl : torch::nn::Module {
  torch::nn::Sequential base{nullptr};

  PulsarValueImpl();

  torch::Tensor forward(torch::Tensor obs, torch::Tensor goal);
};
TORCH_MODULE(PulsarValue);

// ============================================================================
// PulsarQuasimetricCritic  —  sprint5 QuasimetricCritic
// ============================================================================
// phi : [obs|action] → L2-normed repr32
// psi : goal         → L2-normed repr32
// score = -softplus(phi - psi).sum(-1)   (ASYMMETRIC; d(s,a,g) ≠ d(g,s,a))
//
// Named params:  phi.0.*  phi.2.*  psi.0.*  psi.2.*

struct PulsarQuasimetricCriticImpl : torch::nn::Module {
  torch::nn::Sequential phi{nullptr};
  torch::nn::Sequential psi{nullptr};

  PulsarQuasimetricCriticImpl();

  // Returns critic score [B] = -softplus(phi - psi).sum(-1)
  torch::Tensor forward(torch::Tensor obs, torch::Tensor action, torch::Tensor goal);

  // Expose embeddings for InfoNCE loss computation.
  torch::Tensor phi_embed(torch::Tensor obs, torch::Tensor action);
  torch::Tensor psi_embed(torch::Tensor goal);
};
TORCH_MODULE(PulsarQuasimetricCritic);

// ============================================================================
// Symmetric pairwise InfoNCE loss  —  sprint5 compute_pairwise_infonce_loss
// ============================================================================
// sprint5:512-523.  Both row and col directions, plus logsumexp penalty.
torch::Tensor compute_pairwise_infonce_loss(
    PulsarQuasimetricCritic& critic,
    torch::Tensor obs,
    torch::Tensor action,
    torch::Tensor goal,
    double temperature   = 0.1,
    double penalty_coeff = 0.01);

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
