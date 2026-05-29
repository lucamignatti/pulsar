#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>
#include "pulsar/config/config.hpp"
#include "pulsar/model/actor.hpp"
#include "pulsar/model/rssm.hpp"
#include "pulsar/training/rollout_storage.hpp"

namespace pulsar {

struct WorldModelLosses {
  torch::Tensor kl_loss;
  torch::Tensor consistency_loss;
  torch::Tensor icm_forward_loss;
  torch::Tensor icm_inverse_loss;
  torch::Tensor goal_head_loss;
  torch::Tensor total;
  // Scalar metrics for logging
  double kl_loss_val = 0.0;
  double consistency_loss_val = 0.0;
  double icm_forward_loss_val = 0.0;
  double icm_inverse_loss_val = 0.0;
  double goal_head_loss_val = 0.0;
};

class WorldModelTrainer {
 public:
  explicit WorldModelTrainer(const WorldModelConfig& config, const torch::Device& device);

  // Compute all RSSM losses from a rollout. Returns per-step intrinsic rewards [T, agents].
  // Fills losses struct for logging and backward.
  torch::Tensor compute_losses(
      RSSMWorldModel& rssm,
      const RolloutStorage& rollout,
      const ExperimentConfig& config,
      WorldModelLosses& losses_out);

  // Generate imagined goals from terminal latent states for GoalCritic HER augmentation.
  // Returns [N, 4] tensor of reliable imagined goals (or undefined if none pass the gate).
  torch::Tensor imagined_her_goals(
      RSSMWorldModel& rssm,
      const Actor& actor,
      const torch::Tensor& terminal_latents,   // [batch, full_latent_dim]
      const torch::Tensor& terminal_obs,       // [batch, obs_dim]
      const torch::Tensor& action_masks,       // [batch, action_dim]
      int horizon,
      float icm_threshold);

 private:
  static torch::Tensor kl_divergence_free_bits(
      const torch::Tensor& post_mean,
      const torch::Tensor& post_logvar,
      const torch::Tensor& prior_mean,
      const torch::Tensor& prior_logvar,
      float free_bits);

  WorldModelConfig config_;
  torch::Device device_;
};

inline float compute_intrinsic_beta(int update_index, const WorldModelConfig& cfg) {
  if (cfg.intrinsic_anneal_steps <= 0.0F) return 0.0F;
  const float t = static_cast<float>(update_index) / cfg.intrinsic_anneal_steps;
  return cfg.intrinsic_reward_scale * std::max(0.0F, 1.0F - t);
}

}  // namespace pulsar

#endif
