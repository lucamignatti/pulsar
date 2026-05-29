#include "pulsar/training/sac_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

namespace pulsar {

SACTrainer::SACTrainer(const SACCriticConfig& config, const torch::Device& device)
    : config_(config), device_(device) {}

void SACTrainer::init_optimizer(SACCritic& critic) {
  if (optimizer_initialized_) return;
  std::vector<at::Tensor> params;
  // Only online networks (q1_, q2_) have gradients; target networks do not.
  for (const auto& named : critic->named_parameters(true)) {
    // Skip target network parameters (names contain "target")
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

  // Target: r + γ*(1-done)*V_target(s', g)
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

// Compute discounted partial returns G[i,j] = Σ_{t=i}^{j-1} γ^{t-i} * r_t
// for all pairs i < j in [0, L). Returns [L, L] tensor (lower triangle valid).
// Masks out across episode boundaries.
static torch::Tensor compute_partial_returns_tensor(
    const torch::Tensor& rewards,       // [L]
    const torch::Tensor& episode_ends,  // [L]  (1 at boundary start)
    float gamma,
    const torch::Device& device) {
  const int L = static_cast<int>(rewards.size(0));
  // G[i, j] = G[i, j-1] + γ^{j-1-i} * r[j-1]  for j > i
  // Build on CPU, small matrix
  const torch::Tensor r = rewards.to(torch::kCPU).to(torch::kFloat32);
  const torch::Tensor ep = episode_ends.to(torch::kCPU).to(torch::kFloat32);

  torch::Tensor G = torch::zeros({L, L}, torch::TensorOptions().dtype(torch::kFloat32));
  for (int i = 0; i < L; ++i) {
    float discount = 1.0F;
    bool cut = false;
    for (int j = i + 1; j < L; ++j) {
      // Check boundary at j (episode_ends[j]==1 means new episode starts at j)
      if (ep[j].item<float>() > 0.5F) {
        cut = true;
      }
      if (cut) break;
      G[i][j] = (j == i + 1)
          ? discount * r[i].item<float>()
          : G[i][j - 1].item<float>() + discount * r[j - 1].item<float>();
      discount *= gamma;
    }
  }
  return G.to(device);
}

torch::Tensor SACTrainer::compute_lql_losses(
    SACCritic& critic,
    const ReplayBuffer::SegmentBatch& batch) {

  const int L = static_cast<int>(batch.obs.size(0));
  const int B = static_cast<int>(batch.obs.size(1));

  // Process all batch elements with shared Q computation for efficiency.
  // Reshape [L, B, ...] → [L*B, ...]
  const torch::Tensor obs_flat    = batch.obs.reshape({L * B, -1}).to(device_);
  const torch::Tensor act_flat    = batch.actions.reshape({L * B}).to(device_);
  const torch::Tensor goal_flat   = batch.goals.reshape({L * B, -1}).to(device_);

  // Online Q values: [L*B]
  auto [q1_flat, q2_flat] = critic->forward(obs_flat, act_flat, goal_flat);
  const torch::Tensor q_online_flat = torch::minimum(q1_flat, q2_flat);  // [L*B]
  const torch::Tensor q_online = q_online_flat.reshape({L, B});          // [L, B]

  // Target V values: [L*B]
  torch::Tensor v_target_flat;
  {
    torch::NoGradGuard no_grad;
    v_target_flat = critic->value_target(obs_flat, goal_flat);  // [L*B]
  }
  const torch::Tensor v_target = v_target_flat.detach().reshape({L, B});  // [L, B]

  torch::Tensor lb_total = torch::zeros({}, torch::TensorOptions().device(device_));
  torch::Tensor ub_total = torch::zeros({}, torch::TensorOptions().device(device_));
  int pair_count = 0;

  for (int b = 0; b < B; ++b) {
    const torch::Tensor r_b  = batch.rewards.select(1, b);       // [L]
    const torch::Tensor ep_b = batch.episode_ends.select(1, b);  // [L]
    const torch::Tensor G    = compute_partial_returns_tensor(r_b, ep_b, config_.gamma, device_);

    const torch::Tensor q_b = q_online.select(1, b);   // [L]
    const torch::Tensor v_b = v_target.select(1, b);   // [L]

    float discount_ij = config_.gamma;
    for (int i = 0; i < L - 1; ++i) {
      discount_ij = config_.gamma;
      bool cut = false;
      for (int j = i + 1; j < L; ++j) {
        if (ep_b[j].item<float>() > 0.5F) { cut = true; }
        if (cut) break;

        // Lower-bound: G_{i:j} + γ^{j-i} * V_target(s_j) - Q(s_i, a_i)
        const torch::Tensor lb_err = G[i][j] + discount_ij * v_b[j] - q_b[i];
        lb_total = lb_total + torch::relu(lb_err).square();

        // Upper-bound: Q(s_j, a_j) - (V_target(s_i) - G_{i:j}) / γ^{j-i}
        const torch::Tensor ub_err = q_b[j] - (v_b[i] - G[i][j]) / discount_ij;
        ub_total = ub_total + torch::relu(ub_err).square();

        ++pair_count;
        discount_ij *= config_.gamma;
      }
    }

    // Same-state upper-bound: Q(s_i, a_i) ≤ V_target(s_i)
    const torch::Tensor ss_err = torch::relu(q_b - v_b);
    ub_total = ub_total + ss_err.square().sum();
    pair_count += L;
  }

  if (pair_count == 0) return torch::zeros({}, torch::TensorOptions().device(device_));
  return (config_.lql_lambda_lb * lb_total + config_.lql_lambda_ub * ub_total)
      / static_cast<float>(pair_count);
}

SACTrainerLosses SACTrainer::update(
    SACCritic& critic,
    const ReplayBuffer::SegmentBatch& batch) {

  init_optimizer(critic);

  const int L = static_cast<int>(batch.obs.size(0));
  const int B = static_cast<int>(batch.obs.size(1));

  // Flatten for TD loss
  const torch::Tensor obs_flat  = batch.obs.reshape({L * B, -1}).to(device_);
  const torch::Tensor act_flat  = batch.actions.reshape({L * B}).to(device_);
  const torch::Tensor rew_flat  = batch.rewards.reshape({L * B}).to(device_);
  const torch::Tensor next_flat = batch.next_obs.reshape({L * B, -1}).to(device_);
  const torch::Tensor done_flat = batch.dones.reshape({L * B}).to(device_);
  const torch::Tensor goal_flat = batch.goals.reshape({L * B, -1}).to(device_);

  optimizer_->zero_grad();

  const torch::Tensor td   = compute_td_loss(critic, obs_flat, act_flat, rew_flat, next_flat, done_flat, goal_flat);
  const torch::Tensor lql  = compute_lql_losses(critic, batch);
  const torch::Tensor total = td + lql;

  total.backward();
  torch::nn::utils::clip_grad_norm_(critic->parameters(), 1.0F);
  optimizer_->step();

  SACTrainerLosses losses;
  losses.td_loss    = td.item<double>();
  losses.lql_ub_loss = 0.0;  // combined into lql
  losses.lql_lb_loss = lql.item<double>();
  losses.total_loss = total.item<double>();

  {
    torch::NoGradGuard no_grad;
    const int sample_n = std::min(L * B, 512);
    auto [q1s, q2s] = critic->forward(
        obs_flat.narrow(0, 0, sample_n),
        act_flat.narrow(0, 0, sample_n),
        goal_flat.narrow(0, 0, sample_n));
    const torch::Tensor qs = torch::minimum(q1s, q2s);
    losses.mean_q_value = qs.mean().item<double>();
    losses.max_q_value  = qs.abs().max().item<double>();
  }

  return losses;
}

}  // namespace pulsar

#endif
