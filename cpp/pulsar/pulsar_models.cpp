// Pulsar — Phase 2: model implementations.
// See pulsar_models.hpp for the spec.
#include "pulsar/pulsar_models.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>

namespace pulsar {

// layer_init: orthogonal weight, constant bias.  sprint5:452-455.
static torch::nn::Linear layer_init(torch::nn::Linear layer, double std = std::sqrt(2.0),
                                    double bias_const = 0.0) {
  torch::nn::init::orthogonal_(layer->weight, std);
  torch::nn::init::constant_(layer->bias, bias_const);
  return layer;
}

// ============================================================================
// PulsarActor
// ============================================================================

PulsarActorImpl::PulsarActorImpl() {
  // sprint5:460-463
  base = register_module("base", torch::nn::Sequential(
      layer_init(torch::nn::Linear(kObsDim, kHiddenDim)),  // index 0 → "base.0.*"
      torch::nn::Tanh(),                                    // index 1
      layer_init(torch::nn::Linear(kHiddenDim, kHiddenDim)), // index 2 → "base.2.*"
      torch::nn::Tanh()                                     // index 3
  ));

  // policy head: std=0.01  (sprint5:464)
  actor_mean = register_module("actor_mean",
      layer_init(torch::nn::Linear(kHiddenDim, kActionDim), 0.01));

  // actor_logstd: learnable parameter, init zeros  (sprint5:465)
  actor_logstd = register_parameter("actor_logstd",
      torch::zeros({1, kActionDim}));

  // goal projection: small normal init  (sprint5:466-469)
  goal_proj = register_module("goal_proj",
      torch::nn::Linear(kGoalDim, kHiddenDim));
  torch::nn::init::normal_(goal_proj->weight, 0.0, 0.1);
  torch::nn::init::zeros_(goal_proj->bias);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PulsarActorImpl::forward(torch::Tensor obs, torch::Tensor goal) {
  auto features = base->forward(obs);                       // [B, 128]
  if (goal.defined() && goal.numel() > 0) {
    features = features + goal_proj(goal.to(features.dtype())); // sprint5:473-474
  }
  auto mean = actor_mean(features);                          // [B, 8]
  auto std  = torch::exp(actor_logstd.expand_as(mean));     // [B, 8]
  return {mean, std, features};
}

// ============================================================================
// PulsarValue
// ============================================================================

PulsarValueImpl::PulsarValueImpl() {
  // sprint5:481-485. Input = [obs|goal] = 47+6 = 53 dims.
  base = register_module("base", torch::nn::Sequential(
      layer_init(torch::nn::Linear(kObsDim + kGoalDim, kHiddenDim)), // "base.0.*"
      torch::nn::Tanh(),
      layer_init(torch::nn::Linear(kHiddenDim, kHiddenDim)),         // "base.2.*"
      torch::nn::Tanh(),
      layer_init(torch::nn::Linear(kHiddenDim, 1), 1.0)             // "base.4.*" std=1.0
  ));
}

torch::Tensor PulsarValueImpl::forward(torch::Tensor obs, torch::Tensor goal) {
  // sprint5:487
  return base->forward(torch::cat({obs, goal}, /*dim=*/-1));
}

// ============================================================================
// PulsarQuasimetricCritic
// ============================================================================

PulsarQuasimetricCriticImpl::PulsarQuasimetricCriticImpl() {
  // phi: [obs|action] → repr32, no activation on last layer  (sprint5:493-495)
  // input = 47 + 8 = 55 dims
  phi = register_module("phi", torch::nn::Sequential(
      layer_init(torch::nn::Linear(kObsDim + kActionDim, kHiddenDim)), // "phi.0.*"
      torch::nn::Tanh(),
      layer_init(torch::nn::Linear(kHiddenDim, kReprDim))              // "phi.2.*"
  ));

  // psi: goal → repr32  (sprint5:496-498)
  // input = 6 dims (goal)
  psi = register_module("psi", torch::nn::Sequential(
      layer_init(torch::nn::Linear(kGoalDim, kHiddenDim)),  // "psi.0.*"
      torch::nn::Tanh(),
      layer_init(torch::nn::Linear(kHiddenDim, kReprDim))   // "psi.2.*"
  ));
}

torch::Tensor PulsarQuasimetricCriticImpl::phi_embed(torch::Tensor obs, torch::Tensor action) {
  // sprint5:503-504
  auto e = phi->forward(torch::cat({obs, action}, -1));
  return torch::nn::functional::normalize(e, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
}

torch::Tensor PulsarQuasimetricCriticImpl::psi_embed(torch::Tensor goal) {
  // sprint5:505
  auto e = psi->forward(goal);
  return torch::nn::functional::normalize(e, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
}

torch::Tensor PulsarQuasimetricCriticImpl::forward(
    torch::Tensor obs, torch::Tensor action, torch::Tensor goal) {
  const auto phi_out = phi_embed(obs, action);  // [B, 32], L2-normed
  const auto psi_out = psi_embed(goal);          // [B, 32], L2-normed
  // ASYMMETRIC: softplus(phi - psi) ≠ softplus(psi - phi)
  // sprint5:506-507
  const auto d = torch::nn::functional::softplus(phi_out - psi_out).sum(-1);
  return -d;  // [B]  (higher = closer to goal)
}

// ============================================================================
// Symmetric pairwise InfoNCE loss
// ============================================================================
// sprint5:512-523. Logits = -distance / temperature on the [B,B] pairwise matrix.
// Positive pair: (i,i).  Row loss + col loss + logsumexp penalty on both.

torch::Tensor compute_pairwise_infonce_loss(
    PulsarQuasimetricCritic& critic,
    torch::Tensor obs, torch::Tensor action, torch::Tensor goal,
    double temperature, double penalty_coeff) {
  const int64_t B = obs.size(0);
  const auto phi_out = critic->phi_embed(obs, action);   // [B, 32]
  const auto psi_out = critic->psi_embed(goal);           // [B, 32]

  // Pairwise distances [B, B]: phi[i] vs psi[j]  (sprint5:516-518)
  // phi_out.unsqueeze(1) - psi_out.unsqueeze(0) → [B, B, 32]
  const auto distances = torch::nn::functional::softplus(
      phi_out.unsqueeze(1) - psi_out.unsqueeze(0)).sum(-1); // [B, B]

  const double safe_temp = std::max(temperature, 1e-4);
  const auto logits = -distances / safe_temp;              // [B, B]

  // Diagonal indices for positive pairs
  const auto diag_idx = torch::arange(B, obs.options().dtype(torch::kLong));

  // Row and column InfoNCE losses  (sprint5:519-520)
  const auto loss_row = -torch::log_softmax(logits, 1).index({diag_idx, diag_idx}).mean();
  const auto loss_col = -torch::log_softmax(logits, 0).index({diag_idx, diag_idx}).mean();

  // Logsumexp penalty  (sprint5:521-522)
  const auto lse_row = torch::logsumexp(logits, 1).square().mean();
  const auto lse_col = torch::logsumexp(logits, 0).square().mean();
  const auto penalty = penalty_coeff * (lse_row + lse_col);

  return loss_row + loss_col + penalty;
}

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
