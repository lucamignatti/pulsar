#pragma once

#ifdef PULSAR_HAS_TORCH

#include <vector>
#include <torch/torch.h>
#include "pulsar/config/config.hpp"

namespace pulsar {

struct RSSMState {
  torch::Tensor h;  // [batch, latent_dim]
  torch::Tensor z;  // [batch, stochastic_dim]
};

struct RSSMStepOutput {
  torch::Tensor latent;            // [batch, latent_dim + stochastic_dim]
  torch::Tensor h;                 // [batch, latent_dim]
  torch::Tensor z;                 // [batch, stochastic_dim]
  torch::Tensor prior_mean;        // [batch, stochastic_dim]
  torch::Tensor prior_logvar;      // [batch, stochastic_dim]
  torch::Tensor posterior_mean;    // [batch, stochastic_dim] — valid only when obs provided
  torch::Tensor posterior_logvar;  // [batch, stochastic_dim]
  torch::Tensor icm_forward_pred;  // [batch, latent_dim + stochastic_dim]
};

class RSSMWorldModelImpl : public torch::nn::Module {
 public:
  RSSMWorldModelImpl(const WorldModelConfig& config, int obs_dim, int action_dim);

  // Posterior update (training — obs available)
  RSSMStepOutput forward_step(
      const torch::Tensor& obs,
      const torch::Tensor& action,
      const RSSMState& prev_state);

  // Prior-only step (imagination — no obs)
  RSSMStepOutput imagine_step(
      const torch::Tensor& action,
      const RSSMState& prev_state);

  // Full sequence for TBPTT. goal_seq is used to train the goal head.
  // Returns one RSSMStepOutput per timestep.
  // If initial_state is provided, uses it as the starting state instead of zero_state.
  std::vector<RSSMStepOutput> forward_sequence(
      const torch::Tensor& obs_seq,         // [T, batch, obs_dim]
      const torch::Tensor& action_seq,      // [T, batch] int64
      const torch::Tensor& episode_starts,  // [T, batch] float
      const torch::Tensor& goal_seq,        // [T, batch, goal_dim]
      const RSSMState* initial_state = nullptr);

  // goal head: latent → 4D goal prediction
  torch::Tensor predict_goal(const torch::Tensor& latent);

  // ICM inverse model: [latent_t; latent_{t+1}] → action logits
  torch::Tensor inverse_model_forward(
      const torch::Tensor& latent_t,
      const torch::Tensor& latent_tp1);

  [[nodiscard]] int full_latent_dim() const;
  [[nodiscard]] RSSMState zero_state(int64_t batch, const torch::Device& device) const;
  [[nodiscard]] RSSMState split_latent(const torch::Tensor& latent) const;

 private:
  torch::Tensor reparameterize(const torch::Tensor& mean, const torch::Tensor& logvar);
  torch::Tensor make_action_onehot(const torch::Tensor& action) const;

  WorldModelConfig config_;
  int obs_dim_;
  int action_dim_;

  // Own obs encoder — no shared params with Mamba2
  torch::nn::Sequential obs_encoder_{nullptr};         // obs → obs_encoder_dim
  torch::nn::GRUCell gru_{nullptr};                    // [z; a_onehot] → h
  torch::nn::Linear prior_head_{nullptr};              // h → 2*stochastic_dim
  torch::nn::Linear posterior_head_{nullptr};          // [h; obs_enc] → 2*stochastic_dim
  torch::nn::Linear icm_forward_head_{nullptr};        // [latent; a_onehot] → full_latent_dim
  torch::nn::Sequential icm_inverse_head_{nullptr};    // [latent_t; latent_{t+1}] → action_dim
  torch::nn::Linear goal_head_{nullptr};               // latent → 4
};

TORCH_MODULE(RSSMWorldModel);

}  // namespace pulsar

#endif
