#pragma once

#ifdef PULSAR_HAS_TORCH

#include <string>
#include <unordered_map>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/model/vrpo_actor.hpp"

namespace pulsar {

void seed_everything(std::uint64_t seed);

torch::Tensor apply_action_mask_to_logits(const torch::Tensor& logits, const torch::Tensor& action_masks);
torch::Tensor sample_masked_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs = nullptr,
    float temperature = 1.0F);
torch::Tensor masked_action_entropy(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    float temperature = 1.0F);

// Standard GAE baseline
torch::Tensor compute_gae(
    const torch::Tensor& values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_values = {},
    const torch::Tensor& bootstrap_mask = {},
    const torch::Tensor& bootstrap_values = {});

// Q-boosted GAE formulation (Variance-Reduced Expected SARSA(lambda) trace) from Fan & Farina (2026)
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

// Adaptive Centered Expected SARSA(lambda) trace: identical recursion to compute_q_boosted_gae
// except the TD residual subtracts V(s) instead of Q(s, a_t), centering the trace around the
// action-independent baseline. The final advantage still adds the single-step Q(s,a) - V(s) gap
// so the actor receives a low-variance multi-step trace plus the policy-weighted action advantage.
torch::Tensor compute_centered_expected_sarsa_gae(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q = {},
    const torch::Tensor& bootstrap_mask = {},
    const torch::Tensor& bootstrap_v_from_q = {});

// Renamed clipped VRPO policy objective
torch::Tensor clipped_vrpo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range);
torch::Tensor clipped_vrpo_policy_loss(
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

torch::Tensor compute_sparse_event_soon_targets(
    const torch::Tensor& sparse_events,
    const torch::Tensor& dones,
    const torch::Tensor& horizons);

torch::Tensor compute_pairwise_negative_l2_logits(
    const torch::Tensor& lhs_embeddings,
    const torch::Tensor& rhs_embeddings,
    float temperature = 1.0F);

// Symmetric InfoNCE auxiliary goal critic loss from Nimonkar et al. (2025)
torch::Tensor compute_symmetric_infonce_loss(
    const torch::Tensor& logits,
    float logsumexp_penalty_coeff);

// Discrete policy KL divergence for EGGROLL update penalty
float compute_discrete_policy_kl(
    const torch::Tensor& base_logits,
    const torch::Tensor& perturbed_logits,
    const torch::Tensor& action_masks);

}  // namespace pulsar

#endif
