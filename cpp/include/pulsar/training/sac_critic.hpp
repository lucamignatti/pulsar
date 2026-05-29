#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>
#include "pulsar/config/config.hpp"

namespace pulsar {

// Goal-conditioned double-Q critic for off-policy learning.
// Input: [obs; action_onehot; goal] → scalar Q-value.
// Two networks for double-Q (min for target, reduces overestimation).
// No policy — PPO is the policy. SAC is used purely as a critic training framework.
class SACCriticImpl : public torch::nn::Module {
 public:
  SACCriticImpl(const SACCriticConfig& config, int obs_dim, int action_dim, int goal_dim);

  // Returns [Q1(s,a,g), Q2(s,a,g)] — each [batch]
  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& obs,     // [batch, obs_dim]
      const torch::Tensor& action,  // [batch] int64
      const torch::Tensor& goal);   // [batch, goal_dim]

  // V(s, g) = E_π[Q_min(s,·,g)] with soft policy π(a|s) ∝ exp(Q_min(s,a,g) / T).
  // Implemented as a single batched forward pass over all action_dim actions.
  torch::Tensor value(
      const torch::Tensor& obs,    // [batch, obs_dim]
      const torch::Tensor& goal);  // [batch, goal_dim]

  // Target network variants (used for TD backup)
  std::pair<torch::Tensor, torch::Tensor> forward_target(
      const torch::Tensor& obs,
      const torch::Tensor& action,
      const torch::Tensor& goal);

  torch::Tensor value_target(
      const torch::Tensor& obs,
      const torch::Tensor& goal);

  // Soft update: target = tau * online + (1-tau) * target
  void update_target(float tau);

 private:
  torch::nn::Sequential build_q_net() const;
  torch::Tensor encode_input(
      const torch::Tensor& obs,
      const torch::Tensor& action,
      const torch::Tensor& goal) const;

  SACCriticConfig config_;
  int obs_dim_;
  int action_dim_;
  int goal_dim_;

  torch::nn::Sequential q1_{nullptr};
  torch::nn::Sequential q2_{nullptr};
  torch::nn::Sequential q1_target_{nullptr};
  torch::nn::Sequential q2_target_{nullptr};
};

TORCH_MODULE(SACCritic);

}  // namespace pulsar

#endif
