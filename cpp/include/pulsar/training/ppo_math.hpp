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
    const torch::Tensor& next_values = {});
torch::Tensor clipped_ppo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range);
torch::Tensor distributional_value_loss(
    const torch::Tensor& value_logits,
    const torch::Tensor& returns,
    float v_min,
    float v_max,
    int num_atoms);
torch::Tensor sample_quantile_value(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support);
torch::Tensor compute_mean_value(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support);
torch::Tensor compute_distribution_variance(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support);
torch::Tensor compute_distribution_entropy(
    const torch::Tensor& value_logits);
float compute_adaptive_epsilon(
    const torch::Tensor& variance,
    float epsilon_base,
    float epsilon_beta,
    float epsilon_min,
    float epsilon_max);
torch::Tensor compute_confidence_weights(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support,
    const std::string& weight_type,
    float weight_delta,
    bool normalize);

torch::Tensor normalize_advantage(const torch::Tensor& advantages, const torch::Tensor& active_mask);

torch::Tensor compute_finite_horizon_goal_occupancy(
    const torch::Tensor& goal_distances,
    const torch::Tensor& dones,
    float gamma_g,
    float goal_value,
    float kernel_sigma,
    int horizon_H);

torch::Tensor compute_goal_actor_loss_discrete(
    const torch::Tensor& policy_logits,
    const torch::Tensor& action_masks,
    const torch::Tensor& goal_critic_logits,
    const torch::Tensor& goal_atom_support);

float compute_discrete_policy_kl(
    const torch::Tensor& base_logits,
    const torch::Tensor& perturbed_logits,
    const torch::Tensor& action_masks);

float compute_goal_value_correlation(
    const torch::Tensor& predicted_values,
    const torch::Tensor& actual_values);

ContinuumState detach_state(ContinuumState state);
ContinuumState clone_state(const ContinuumState& state);
ContinuumState state_to_device(ContinuumState state, const torch::Device& device);
ContinuumState gather_state(const ContinuumState& state, const torch::Tensor& indices);
void scatter_state(ContinuumState& dst, const torch::Tensor& indices, const ContinuumState& src);

}  // namespace pulsar

#endif
