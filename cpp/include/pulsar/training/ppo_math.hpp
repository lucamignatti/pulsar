#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/model/actor_critic.hpp"

namespace pulsar {

torch::Tensor apply_action_mask_to_logits(const torch::Tensor& logits, const torch::Tensor& action_masks);
torch::Tensor sample_masked_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs = nullptr);
torch::Tensor masked_action_entropy(const torch::Tensor& logits, const torch::Tensor& action_masks);
torch::Tensor categorical_value_projection(const torch::Tensor& returns, float v_min, float v_max, int num_atoms);
torch::Tensor compute_confidence_weights(
    const SharedActorCritic& model,
    const PPOConfig& config,
    const torch::Tensor& value_logits);
torch::Tensor compute_adaptive_epsilon(
    const SharedActorCritic& model,
    const PPOConfig& config,
    const torch::Tensor& value_logits);
void validate_precision_mode_or_throw(const PPOConfig::PrecisionConfig& precision, const torch::Device& device);

}  // namespace pulsar

#endif
