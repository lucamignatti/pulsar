#pragma once

#ifdef PULSAR_HAS_TORCH

#include <string>
#include <unordered_map>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/model/ppo_actor.hpp"

namespace pulsar {

void seed_everything(std::uint64_t seed);

torch::Tensor apply_action_mask_to_logits(const torch::Tensor& logits, const torch::Tensor& action_masks);
torch::Tensor sample_masked_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs = nullptr);
torch::Tensor masked_action_entropy(const torch::Tensor& logits, const torch::Tensor& action_masks);
torch::Tensor compute_gae(
    const torch::Tensor& values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_values = {},
    const torch::Tensor& bootstrap_mask = {},
    const torch::Tensor& bootstrap_values = {});
torch::Tensor compute_q_boosted_gae(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q = {},
    const torch::Tensor& bootstrap_mask = {},
    const torch::Tensor& bootstrap_v_from_q = {});
torch::Tensor clipped_ppo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range);
torch::Tensor clipped_ppo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    const torch::Tensor& clip_range);

torch::Tensor normalize_advantage(const torch::Tensor& advantages, const torch::Tensor& active_mask);

torch::Tensor sample_future_goal_positions(
    const torch::Tensor& goal_positions,
    const torch::Tensor& dones,
    const torch::Tensor& episode_starts,
    int max_future);

torch::Tensor compute_pairwise_negative_l2_logits(
    const torch::Tensor& lhs_embeddings,
    const torch::Tensor& rhs_embeddings);

torch::Tensor compute_symmetric_infonce_loss(
    const torch::Tensor& logits,
    float logsumexp_penalty_coeff);

float compute_discrete_policy_kl(
    const torch::Tensor& base_logits,
    const torch::Tensor& perturbed_logits,
    const torch::Tensor& action_masks);

}  // namespace pulsar

#endif
