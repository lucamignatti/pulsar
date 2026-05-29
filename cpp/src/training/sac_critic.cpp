#include "pulsar/training/sac_critic.hpp"

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

namespace pulsar {

SACCriticImpl::SACCriticImpl(
    const SACCriticConfig& config,
    int obs_dim,
    int action_dim,
    int goal_dim)
    : config_(config), obs_dim_(obs_dim), action_dim_(action_dim), goal_dim_(goal_dim) {

  q1_        = register_module("q1",        build_q_net());
  q2_        = register_module("q2",        build_q_net());
  q1_target_ = register_module("q1_target", build_q_net());
  q2_target_ = register_module("q2_target", build_q_net());

  // Initialize target networks to match online networks and disable their gradients
  {
    torch::NoGradGuard no_grad;
    auto src1 = q1_->parameters();
    auto dst1 = q1_target_->parameters();
    for (std::size_t i = 0; i < src1.size() && i < dst1.size(); ++i) {
      dst1[i].copy_(src1[i]);
    }
    auto src2 = q2_->parameters();
    auto dst2 = q2_target_->parameters();
    for (std::size_t i = 0; i < src2.size() && i < dst2.size(); ++i) {
      dst2[i].copy_(src2[i]);
    }
  }
  for (auto& p : q1_target_->parameters()) p.set_requires_grad(false);
  for (auto& p : q2_target_->parameters()) p.set_requires_grad(false);
}

torch::nn::Sequential SACCriticImpl::build_q_net() const {
  // Input: obs_dim + action_dim (one-hot) + goal_dim → hidden → ... → 1
  const int input_dim = obs_dim_ + action_dim_ + goal_dim_;
  torch::nn::Sequential net;
  net->push_back(torch::nn::Linear(input_dim, config_.hidden_dim));
  net->push_back(torch::nn::SiLU());
  for (int l = 1; l < config_.num_layers; ++l) {
    net->push_back(torch::nn::Linear(config_.hidden_dim, config_.hidden_dim));
    net->push_back(torch::nn::SiLU());
  }
  net->push_back(torch::nn::Linear(config_.hidden_dim, 1));
  return net;
}

torch::Tensor SACCriticImpl::encode_input(
    const torch::Tensor& obs,
    const torch::Tensor& action,
    const torch::Tensor& goal) const {
  const int64_t batch = obs.size(0);
  torch::Tensor a_onehot = torch::zeros(
      {batch, action_dim_},
      torch::TensorOptions().dtype(torch::kFloat32).device(obs.device()));
  a_onehot.scatter_(1, action.unsqueeze(1).to(torch::kLong), 1.0F);
  return torch::cat({obs.to(torch::kFloat32), a_onehot, goal.to(torch::kFloat32)}, -1);
}

std::pair<torch::Tensor, torch::Tensor> SACCriticImpl::forward(
    const torch::Tensor& obs,
    const torch::Tensor& action,
    const torch::Tensor& goal) {
  const torch::Tensor x = encode_input(obs, action, goal);
  const torch::Tensor q1 = q1_->forward(x).squeeze(-1);
  const torch::Tensor q2 = q2_->forward(x).squeeze(-1);
  return {q1, q2};
}

std::pair<torch::Tensor, torch::Tensor> SACCriticImpl::forward_target(
    const torch::Tensor& obs,
    const torch::Tensor& action,
    const torch::Tensor& goal) {
  const torch::Tensor x = encode_input(obs, action, goal);
  const torch::Tensor q1 = q1_target_->forward(x).squeeze(-1);
  const torch::Tensor q2 = q2_target_->forward(x).squeeze(-1);
  return {q1, q2};
}

torch::Tensor SACCriticImpl::value(
    const torch::Tensor& obs,
    const torch::Tensor& goal) {
  const int64_t batch = obs.size(0);

  // Tile obs and goal: [batch, obs_dim] → [batch*action_dim, obs_dim]
  const torch::Tensor obs_t = obs.unsqueeze(1)
      .expand({batch, action_dim_, obs.size(1)})
      .reshape({batch * action_dim_, obs.size(1)});
  const torch::Tensor goal_t = goal.unsqueeze(1)
      .expand({batch, action_dim_, goal.size(1)})
      .reshape({batch * action_dim_, goal.size(1)});

  // All actions: [batch*action_dim]
  const torch::Tensor all_actions = torch::arange(action_dim_,
      torch::TensorOptions().dtype(torch::kLong).device(obs.device()))
      .unsqueeze(0).expand({batch, action_dim_}).reshape({batch * action_dim_});

  // Single batched forward pass
  auto [q1, q2] = forward(obs_t, all_actions, goal_t);
  const torch::Tensor q_min = torch::minimum(q1, q2).reshape({batch, action_dim_});  // [batch, action_dim]

  // Soft policy: π(a|s) ∝ exp(Q_min(s,a,g) / T)
  const torch::Tensor pi = torch::softmax(q_min / config_.temperature, /*dim=*/1);  // [batch, action_dim]

  // V = E_π[Q_min]
  return (pi * q_min).sum(1);  // [batch]
}

torch::Tensor SACCriticImpl::value_target(
    const torch::Tensor& obs,
    const torch::Tensor& goal) {
  const int64_t batch = obs.size(0);

  const torch::Tensor obs_t = obs.unsqueeze(1)
      .expand({batch, action_dim_, obs.size(1)})
      .reshape({batch * action_dim_, obs.size(1)});
  const torch::Tensor goal_t = goal.unsqueeze(1)
      .expand({batch, action_dim_, goal.size(1)})
      .reshape({batch * action_dim_, goal.size(1)});
  const torch::Tensor all_actions = torch::arange(action_dim_,
      torch::TensorOptions().dtype(torch::kLong).device(obs.device()))
      .unsqueeze(0).expand({batch, action_dim_}).reshape({batch * action_dim_});

  auto [q1, q2] = forward_target(obs_t, all_actions, goal_t);
  const torch::Tensor q_min = torch::minimum(q1, q2).reshape({batch, action_dim_});
  const torch::Tensor pi = torch::softmax(q_min / config_.temperature, 1);
  return (pi * q_min).sum(1);
}

void SACCriticImpl::update_target(float tau) {
  torch::NoGradGuard no_grad;
  auto online1 = q1_->parameters();
  auto target1 = q1_target_->parameters();
  for (std::size_t i = 0; i < online1.size() && i < target1.size(); ++i) {
    target1[i].mul_(1.0F - tau).add_(online1[i] * tau);
  }
  auto online2 = q2_->parameters();
  auto target2 = q2_target_->parameters();
  for (std::size_t i = 0; i < online2.size() && i < target2.size(); ++i) {
    target2[i].mul_(1.0F - tau).add_(online2[i] * tau);
  }
}

}  // namespace pulsar

#endif
