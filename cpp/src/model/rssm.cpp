#include "pulsar/model/rssm.hpp"

#ifdef PULSAR_HAS_TORCH

#include <stdexcept>
#include <torch/torch.h>

namespace pulsar {

RSSMWorldModelImpl::RSSMWorldModelImpl(
    const WorldModelConfig& config,
    int obs_dim,
    int action_dim)
    : config_(config), obs_dim_(obs_dim), action_dim_(action_dim) {

  // Obs encoder: obs_dim → 256 → obs_encoder_dim (separate from Mamba2)
  obs_encoder_ = register_module("obs_encoder", torch::nn::Sequential(
      torch::nn::Linear(obs_dim, 256),
      torch::nn::SiLU(),
      torch::nn::Linear(256, config.obs_encoder_dim),
      torch::nn::SiLU()));

  // GRU input: [z_{t-1}; a_{t-1}_onehot]
  gru_ = register_module("gru", torch::nn::GRUCell(
      torch::nn::GRUCellOptions(config.stochastic_dim + action_dim, config.latent_dim)));

  // Prior: h → (mean, logvar) of z
  prior_head_ = register_module("prior_head",
      torch::nn::Linear(config.latent_dim, 2 * config.stochastic_dim));

  // Posterior: [h; obs_enc] → (mean, logvar) of z
  posterior_head_ = register_module("posterior_head",
      torch::nn::Linear(config.latent_dim + config.obs_encoder_dim, 2 * config.stochastic_dim));

  // ICM forward: [latent; a_onehot] → next_latent
  icm_forward_head_ = register_module("icm_forward_head",
      torch::nn::Linear(full_latent_dim() + action_dim, full_latent_dim()));

  // ICM inverse: [latent_t; latent_{t+1}] → action logits
  icm_inverse_head_ = register_module("icm_inverse_head", torch::nn::Sequential(
      torch::nn::Linear(2 * full_latent_dim(), 256),
      torch::nn::SiLU(),
      torch::nn::Linear(256, action_dim)));

  // Goal head: latent → 4D goal
  goal_head_ = register_module("goal_head",
      torch::nn::Linear(full_latent_dim(), 4));
}

int RSSMWorldModelImpl::full_latent_dim() const {
  return config_.latent_dim + config_.stochastic_dim;
}

RSSMState RSSMWorldModelImpl::zero_state(int64_t batch, const torch::Device& device) const {
  return {
      torch::zeros({batch, config_.latent_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
      torch::zeros({batch, config_.stochastic_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
  };
}

RSSMState RSSMWorldModelImpl::split_latent(const torch::Tensor& latent) const {
  return {
      latent.narrow(-1, 0, config_.latent_dim),
      latent.narrow(-1, config_.latent_dim, config_.stochastic_dim),
  };
}

torch::Tensor RSSMWorldModelImpl::reparameterize(
    const torch::Tensor& mean,
    const torch::Tensor& logvar) {
  const torch::Tensor std = torch::exp(0.5F * logvar.clamp(-10.0F, 10.0F));
  return mean + std * torch::randn_like(std);
}

torch::Tensor RSSMWorldModelImpl::make_action_onehot(const torch::Tensor& action) const {
  const int64_t batch = action.size(0);
  torch::Tensor onehot = torch::zeros(
      {batch, action_dim_},
      torch::TensorOptions().dtype(torch::kFloat32).device(action.device()));
  onehot.scatter_(1, action.unsqueeze(1), 1.0F);
  return onehot;
}

RSSMStepOutput RSSMWorldModelImpl::forward_step(
    const torch::Tensor& obs,
    const torch::Tensor& action,
    const RSSMState& prev_state) {

  const torch::Tensor a_onehot = make_action_onehot(action);

  // GRU input: [prev_z; prev_a_onehot]
  const torch::Tensor gru_input = torch::cat({prev_state.z, a_onehot}, -1);
  const torch::Tensor h = gru_(gru_input, prev_state.h);

  // Prior: p(z_t | h_t)
  const torch::Tensor prior_params = prior_head_(h);
  const torch::Tensor prior_mean = prior_params.narrow(-1, 0, config_.stochastic_dim);
  const torch::Tensor prior_logvar = prior_params.narrow(-1, config_.stochastic_dim, config_.stochastic_dim);

  // Posterior: q(z_t | h_t, obs_t)
  const torch::Tensor obs_enc = obs_encoder_->forward(obs);
  const torch::Tensor post_params = posterior_head_(torch::cat({h, obs_enc}, -1));
  const torch::Tensor post_mean = post_params.narrow(-1, 0, config_.stochastic_dim);
  const torch::Tensor post_logvar = post_params.narrow(-1, config_.stochastic_dim, config_.stochastic_dim);

  // Sample z from posterior
  const torch::Tensor z = reparameterize(post_mean, post_logvar);
  const torch::Tensor latent = torch::cat({h, z}, -1);

  // ICM forward prediction
  const torch::Tensor icm_input = torch::cat({latent, a_onehot}, -1);
  const torch::Tensor icm_pred = icm_forward_head_(icm_input);

  RSSMStepOutput out;
  out.latent = latent;
  out.h = h;
  out.z = z;
  out.prior_mean = prior_mean;
  out.prior_logvar = prior_logvar;
  out.posterior_mean = post_mean;
  out.posterior_logvar = post_logvar;
  out.icm_forward_pred = icm_pred;
  return out;
}

RSSMStepOutput RSSMWorldModelImpl::imagine_step(
    const torch::Tensor& action,
    const RSSMState& prev_state) {

  const torch::Tensor a_onehot = make_action_onehot(action);

  const torch::Tensor gru_input = torch::cat({prev_state.z, a_onehot}, -1);
  const torch::Tensor h = gru_(gru_input, prev_state.h);

  // Prior only (no obs)
  const torch::Tensor prior_params = prior_head_(h);
  const torch::Tensor prior_mean = prior_params.narrow(-1, 0, config_.stochastic_dim);
  const torch::Tensor prior_logvar = prior_params.narrow(-1, config_.stochastic_dim, config_.stochastic_dim);

  const torch::Tensor z = reparameterize(prior_mean, prior_logvar);
  const torch::Tensor latent = torch::cat({h, z}, -1);

  const torch::Tensor icm_input = torch::cat({latent, a_onehot}, -1);
  const torch::Tensor icm_pred = icm_forward_head_(icm_input);

  RSSMStepOutput out;
  out.latent = latent;
  out.h = h;
  out.z = z;
  out.prior_mean = prior_mean;
  out.prior_logvar = prior_logvar;
  out.icm_forward_pred = icm_pred;
  return out;
}

std::vector<RSSMStepOutput> RSSMWorldModelImpl::forward_sequence(
    const torch::Tensor& obs_seq,
    const torch::Tensor& action_seq,
    const torch::Tensor& episode_starts,
    const torch::Tensor& goal_seq,
    const RSSMState* initial_state) {

  const int64_t T = obs_seq.size(0);
  const int64_t batch = obs_seq.size(1);
  const torch::Device dev = obs_seq.device();

  RSSMState state = (initial_state != nullptr)
      ? *initial_state
      : zero_state(batch, dev);
  std::vector<RSSMStepOutput> outputs;
  outputs.reserve(static_cast<std::size_t>(T));

  for (int64_t t = 0; t < T; ++t) {
    // Reset h and z at episode boundaries (before GRU step)
    if (episode_starts.defined()) {
      const torch::Tensor reset_mask = episode_starts[t].unsqueeze(-1);  // [batch, 1]
      state.h = state.h * (1.0F - reset_mask);
      state.z = state.z * (1.0F - reset_mask);
    }

    const torch::Tensor obs_t = obs_seq[t];
    const torch::Tensor action_t = action_seq[t];

    RSSMStepOutput out = forward_step(obs_t, action_t, state);
    state.h = out.h;
    state.z = out.z;
    outputs.push_back(std::move(out));
  }

  (void)goal_seq;  // goal_seq training is handled externally via predict_goal
  return outputs;
}

torch::Tensor RSSMWorldModelImpl::predict_goal(const torch::Tensor& latent) {
  return goal_head_(latent);
}

torch::Tensor RSSMWorldModelImpl::inverse_model_forward(
    const torch::Tensor& latent_t,
    const torch::Tensor& latent_tp1) {
  return icm_inverse_head_->forward(torch::cat({latent_t, latent_tp1}, -1));
}

}  // namespace pulsar

#endif
