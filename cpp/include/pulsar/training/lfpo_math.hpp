#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/model/latent_future_actor.hpp"

namespace pulsar {

torch::Tensor apply_action_mask_to_logits(const torch::Tensor& logits, const torch::Tensor& action_masks);
torch::Tensor sample_masked_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs = nullptr);
torch::Tensor masked_action_entropy(const torch::Tensor& logits, const torch::Tensor& action_masks);
torch::Tensor latent_action_scores(const torch::Tensor& outcome_logits);
torch::Tensor relative_candidate_advantages(const torch::Tensor& candidate_scores);
torch::Tensor clipped_lfpo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range);
ContinuumState detach_state(ContinuumState state);
ContinuumState clone_state(const ContinuumState& state);
ContinuumState state_to_device(ContinuumState state, const torch::Device& device);
ContinuumState gather_state(const ContinuumState& state, const torch::Tensor& indices);
void scatter_state(ContinuumState& dst, const torch::Tensor& indices, const ContinuumState& src);

}  // namespace pulsar

#endif
