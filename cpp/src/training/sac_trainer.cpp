#include "pulsar/training/sac_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

namespace pulsar {

SACTrainer::SACTrainer(const SACCriticConfig& config, const torch::Device& device)
    : config_(config), device_(device) {}

void SACTrainer::init_optimizer(SACCritic& critic) {
  if (optimizer_initialized_) return;
  std::vector<at::Tensor> params;
  for (const auto& named : critic->named_parameters(true)) {
    if (named.key().find("target") == std::string::npos && named.value().requires_grad()) {
      params.push_back(named.value());
    }
  }
  optimizer_ = std::make_unique<torch::optim::Adam>(
      params,
      torch::optim::AdamOptions(static_cast<double>(config_.learning_rate)));
  optimizer_initialized_ = true;
}

torch::Tensor SACTrainer::compute_td_loss(
    SACCritic& critic,
    const torch::Tensor& obs,
    const torch::Tensor& actions,
    const torch::Tensor& rewards,
    const torch::Tensor& next_obs,
    const torch::Tensor& dones,
    const torch::Tensor& goals) {

  torch::Tensor target_v;
  {
    torch::NoGradGuard no_grad;
    target_v = critic->value_target(next_obs, goals);
  }
  const torch::Tensor target = (rewards + config_.gamma * (1.0F - dones) * target_v).detach();
  auto [q1, q2] = critic->forward(obs, actions, goals);
  return 0.5F * (torch::nn::functional::smooth_l1_loss(q1, target) +
                 torch::nn::functional::smooth_l1_loss(q2, target));
}

// Build discounted partial-return matrix G[L,L] fully vectorized.
// G[i,j] = Σ_{t=i}^{j-1} γ^{t-i} * r[t]   (valid only when valid_pairs[i,j]==1)
// valid_pairs[i,j] = 1 iff j>i AND no episode boundary in (i, j]
//
// Episode boundaries encoded via cumsum: boundary between i and j exists iff
// episode_ends.cumsum()[j] > episode_ends.cumsum()[i]
//
// All computation stays on-device; zero .item() calls.
static std::pair<torch::Tensor, torch::Tensor> build_G_and_valid(
    const torch::Tensor& rewards,       // [L], on device
    const torch::Tensor& episode_ends,  // [L], on device
    float gamma) {

  const int L = static_cast<int>(rewards.size(0));
  const auto dev  = rewards.device();
  const auto opts = rewards.options();

  // Valid-pair mask: no boundary strictly between i and j
  const torch::Tensor ep_cs = episode_ends.cumsum(0);  // [L]
  // valid[i,j] = (ep_cs[j] - ep_cs[i] == 0) AND j > i
  const torch::Tensor ep_i = ep_cs.unsqueeze(1).expand({L, L});   // [L,L]
  const torch::Tensor ep_j = ep_cs.unsqueeze(0).expand({L,L});    // [L,L]
  const torch::Tensor valid_pairs =
      ((ep_j - ep_i) == 0).to(opts.dtype()) *
      torch::ones({L, L}, opts).triu(1);  // [L,L], float

  // Build G column by column — O(L) iterations, each fully vectorized over L.
  // disc[k] = gamma^k
  const torch::Tensor disc = torch::pow(
      torch::tensor(gamma, opts),
      torch::arange(L, opts));  // [L]

  torch::Tensor G = torch::zeros({L, L}, opts);
  for (int j = 1; j < L; ++j) {
    // G[i,j] = G[i,j-1] + gamma^{j-1-i} * r[j-1]   for i in [0..j-1]
    // gamma^{j-1-i} for i=0..j-1 → disc[j-1], disc[j-2], ..., disc[0]
    const torch::Tensor new_col =
        G.narrow(0, 0, j).select(1, j - 1)
        + disc.narrow(0, 0, j).flip(0) * rewards[j - 1];
    G.narrow(0, 0, j).select(1, j).copy_(new_col);
  }

  G = G * valid_pairs;  // zero out invalid pairs
  return {G, valid_pairs};
}

torch::Tensor SACTrainer::compute_lql_losses(
    SACCritic& critic,
    const ReplayBuffer::SegmentBatch& batch) {

  const int L = static_cast<int>(batch.obs.size(0));
  const int B = static_cast<int>(batch.obs.size(1));

  // Flatten [L, B, ...] → [L*B, ...] for a single batched Q/V forward pass.
  const torch::Tensor obs_flat  = batch.obs.reshape({L * B, -1}).to(device_);
  const torch::Tensor act_flat  = batch.actions.reshape({L * B}).to(device_);
  const torch::Tensor goal_flat = batch.goals.reshape({L * B, -1}).to(device_);

  auto [q1_flat, q2_flat] = critic->forward(obs_flat, act_flat, goal_flat);
  const torch::Tensor q_flat = torch::minimum(q1_flat, q2_flat);  // [L*B]
  const torch::Tensor q_lb   = q_flat.reshape({L, B});             // [L, B]

  torch::Tensor v_flat;
  {
    torch::NoGradGuard no_grad;
    v_flat = critic->value_target(obs_flat, goal_flat).detach();   // [L*B]
  }
  const torch::Tensor v = v_flat.reshape({L, B});  // [L, B]

  // Same-state upper-bound across all (L*B) elements: Q(s,a) ≤ V_target(s)
  const torch::Tensor ss_loss = torch::relu(q_flat - v_flat.detach()).square().mean();

  torch::Tensor lb_total = torch::zeros({}, torch::TensorOptions().device(device_));
  torch::Tensor ub_total = torch::zeros({}, torch::TensorOptions().device(device_));
  int pair_count = 0;

  // Build discount-matrix [L,L] once (same for all batch elements).
  const torch::Tensor j_idx = torch::arange(L,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_))
      .unsqueeze(0).expand({L, L});
  const torch::Tensor i_idx = torch::arange(L,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_))
      .unsqueeze(1).expand({L, L});
  // disc_matrix[i,j] = gamma^{j-i} (will be masked by valid_pairs)
  const torch::Tensor disc_matrix_raw =
      torch::pow(config_.gamma, (j_idx - i_idx).clamp_min(0.0F));

  for (int b = 0; b < B; ++b) {
    const torch::Tensor r_b   = batch.rewards.select(1, b).to(device_);       // [L]
    const torch::Tensor ep_b  = batch.episode_ends.select(1, b).to(device_);  // [L]
    const torch::Tensor q_b   = q_lb.select(1, b);   // [L]
    const torch::Tensor v_b   = v.select(1, b);      // [L]

    auto [G, valid] = build_G_and_valid(r_b, ep_b, config_.gamma);  // [L,L]

    // Mask disc_matrix by validity
    const torch::Tensor disc = disc_matrix_raw * valid;  // [L,L]
    const torch::Tensor inv_disc = torch::where(
        disc > 1e-8F,
        1.0F / disc.clamp_min(1e-8F),
        torch::zeros_like(disc));

    // LB: hinge_sq( G[i,j] + γ^{j-i}*V_target(s_j) - Q(s_i,a_i) )
    // Broadcast: V[j] → [1,L], Q[i] → [L,1]
    const torch::Tensor lb_mat = G + disc * v_b.unsqueeze(0) - q_b.unsqueeze(1);
    lb_total = lb_total + (torch::relu(lb_mat) * valid).square().sum();

    // UB: hinge_sq( Q(s_j,a_j) - (V_target(s_i) - G[i,j]) / γ^{j-i} )
    const torch::Tensor ub_mat = q_b.unsqueeze(0) - (v_b.unsqueeze(1) - G) * inv_disc;
    ub_total = ub_total + (torch::relu(ub_mat) * valid).square().sum();

    // Count valid pairs (for normalization)
    pair_count += static_cast<int>(valid.sum().item<float>());
  }

  if (pair_count == 0) {
    return ss_loss;
  }
  const float norm = static_cast<float>(pair_count);
  return ss_loss
      + config_.lql_lambda_lb * lb_total / norm
      + config_.lql_lambda_ub * ub_total / norm;
}

SACTrainerLosses SACTrainer::update(
    SACCritic& critic,
    const ReplayBuffer::SegmentBatch& batch) {

  init_optimizer(critic);

  const int L = static_cast<int>(batch.obs.size(0));
  const int B = static_cast<int>(batch.obs.size(1));

  const torch::Tensor obs_flat  = batch.obs.reshape({L * B, -1}).to(device_);
  const torch::Tensor act_flat  = batch.actions.reshape({L * B}).to(device_);
  const torch::Tensor rew_flat  = batch.rewards.reshape({L * B}).to(device_);
  const torch::Tensor next_flat = batch.next_obs.reshape({L * B, -1}).to(device_);
  const torch::Tensor done_flat = batch.dones.reshape({L * B}).to(device_);
  const torch::Tensor goal_flat = batch.goals.reshape({L * B, -1}).to(device_);

  optimizer_->zero_grad();

  const torch::Tensor td  = compute_td_loss(critic, obs_flat, act_flat, rew_flat,
                                              next_flat, done_flat, goal_flat);
  const torch::Tensor lql = compute_lql_losses(critic, batch);
  const torch::Tensor total = td + lql;

  total.backward();
  torch::nn::utils::clip_grad_norm_(critic->parameters(), 1.0F);
  optimizer_->step();

  SACTrainerLosses losses;
  losses.td_loss    = td.item<double>();
  losses.lql_lb_loss = lql.item<double>();
  losses.total_loss = total.item<double>();

  {
    torch::NoGradGuard no_grad;
    const int sn = std::min(L * B, 512);
    auto [q1s, q2s] = critic->forward(
        obs_flat.narrow(0, 0, sn),
        act_flat.narrow(0, 0, sn),
        goal_flat.narrow(0, 0, sn));
    const torch::Tensor qs = torch::minimum(q1s, q2s);
    losses.mean_q_value = qs.mean().item<double>();
    losses.max_q_value  = qs.abs().max().item<double>();
  }

  return losses;
}

}  // namespace pulsar

#endif
