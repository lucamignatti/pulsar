#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/model/vrpo_actor.hpp"
#include "pulsar/training/gradient_surgery.hpp"

namespace pulsar {

torch::Tensor sample_masked_gumbel_softmax(
    const torch::Tensor& logits,
    const torch::Tensor& masks,
    float temperature = 1.0F);

torch::Tensor goal_actor_critic_loss(
    GoalCritic& goal_critic,
    const torch::Tensor& features,
    const torch::Tensor& logits,
    const torch::Tensor& masks,
    const torch::Tensor& future_goals,
    int contrastive_batch_size);

class GCRLTrainer {
 public:
  GCRLTrainer(
      const GoalCriticConfig& config,
      const torch::Device& device);

  void compute_gcrl_losses(
      GoalCritic& goal_critic,
      const torch::Tensor& active_features,
      const torch::Tensor& active_actions,
      const torch::Tensor& active_logits,
      const torch::Tensor& active_masks,
      const torch::Tensor& active_future_goal_pos,
      bool compute_critic_loss,
      bool compute_actor_loss,
      torch::Tensor& goal_loss,
      torch::Tensor& actor_goal_loss,
      torch::Tensor& goal_score);

 private:
  GoalCriticConfig config_;
  torch::Device device_;
};

}  // namespace pulsar

#endif
