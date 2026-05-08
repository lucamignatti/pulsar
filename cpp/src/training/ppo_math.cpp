#include <algorithm>
#include <cmath>
#include <limits>

#include "pulsar/training/ppo_math.hpp"
#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

void seed_everything(std::uint64_t seed) {
  torch::manual_seed(static_cast<int64_t>(seed));
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed_all(static_cast<int64_t>(seed));
  }
}

namespace {

torch::Tensor sample_categorical_from_logits(const torch::Tensor& logits) {
  const torch::Tensor uniform = torch::rand_like(logits).clamp_min_(1.0e-6);
  const torch::Tensor gumbel = -torch::log(-torch::log(uniform));
  return (logits + gumbel).argmax(-1);
}

torch::Tensor detach_tensor(const torch::Tensor& tensor) {
  return tensor.defined() ? tensor.detach() : tensor;
}

torch::Tensor clone_tensor(const torch::Tensor& tensor) {
  return tensor.defined() ? tensor.detach().clone() : tensor;
}

torch::Tensor tensor_to_device(const torch::Tensor& tensor, const torch::Device& device) {
  return tensor.defined() ? tensor.to(device) : tensor;
}

torch::Tensor gather_tensor(const torch::Tensor& tensor, const torch::Tensor& indices) {
  if (!tensor.defined()) {
    return tensor;
  }
  return tensor.index_select(0, indices.to(tensor.device()));
}

void scatter_tensor(torch::Tensor& dst, const torch::Tensor& indices, const torch::Tensor& src) {
  if (dst.defined() && src.defined()) {
    dst.index_copy_(0, indices.to(dst.device()), src);
  }
}

}  // namespace

torch::Tensor apply_action_mask_to_logits(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  return logits.masked_fill(action_masks.logical_not(), -std::numeric_limits<float>::infinity());
}

torch::Tensor sample_masked_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs) {
  const torch::Tensor masked = apply_action_mask_to_logits(logits, action_masks);
  const torch::Tensor actions = deterministic ? masked.argmax(-1) : sample_categorical_from_logits(masked);
  if (log_probs != nullptr) {
    *log_probs = torch::log_softmax(masked, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
  }
  return actions;
}

torch::Tensor masked_action_entropy(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "entropy");
  const torch::Tensor masked = apply_action_mask_to_logits(logits, action_masks);
  const torch::Tensor probs = torch::softmax(masked, -1);
  const torch::Tensor valid_counts = action_masks.to(torch::kFloat32).sum(-1);
  const torch::Tensor trivial = valid_counts <= 1.0F;
  const torch::Tensor raw_entropy = -(probs * torch::log(probs + 1.0e-8)).sum(-1);
  const torch::Tensor normalized = raw_entropy / valid_counts.log().clamp_min(1.0e-6);
  return torch::where(trivial, torch::zeros_like(normalized), normalized);
}

torch::Tensor compute_gae(
    const torch::Tensor& values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_values) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "compute_gae");
  const int64_t steps = values.size(0);
  const int64_t agents = values.size(1);
  torch::Tensor advantages = torch::zeros({steps, agents}, values.options());
  torch::Tensor last_gae = torch::zeros({agents}, values.options());

  const torch::Tensor boundary_value = next_values.defined()
      ? next_values.to(values.device()).to(values.dtype())
      : torch::zeros({agents}, values.options());

  for (int64_t t = steps - 1; t >= 0; --t) {
    const torch::Tensor next_value = (t < steps - 1) ? values[t + 1] : boundary_value;
    const torch::Tensor non_terminal = 1.0 - dones[t];
    const torch::Tensor delta = rewards[t] + gamma * next_value * non_terminal - values[t];
    last_gae = delta + gamma * gae_lambda * non_terminal * last_gae;
    advantages[t] = last_gae.clone();
  }
  return advantages;
}

torch::Tensor clipped_ppo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "clipped_ppo_loss");
  const torch::Tensor ratio = torch::exp(current_log_probs - old_log_probs);
  const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - clip_range, 1.0 + clip_range);
  return -torch::min(ratio * advantages, clipped_ratio * advantages);
}

torch::Tensor clipped_ppo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    const torch::Tensor& clip_range) {
  const torch::Tensor ratio = torch::exp(current_log_probs - old_log_probs);
  const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - clip_range, 1.0 + clip_range);
  return -torch::min(ratio * advantages, clipped_ratio * advantages);
}

torch::Tensor distributional_value_loss(
    const torch::Tensor& value_logits,
    const torch::Tensor& returns,
    float v_min,
    float v_max,
    int num_atoms) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "distributional_value_loss");
  const float delta_z = (v_max - v_min) / static_cast<float>(num_atoms - 1);
  const torch::Tensor clamped_returns = returns.clamp(v_min, v_max);
  const torch::Tensor b = (clamped_returns - v_min) / delta_z;
  const torch::Tensor lower = b.floor().clamp(0, num_atoms - 1).to(torch::kLong);
  const torch::Tensor upper = b.ceil().clamp(0, num_atoms - 1).to(torch::kLong);
  const torch::Tensor weight_upper = (b - lower.to(torch::kFloat32)).clamp(0.0, 1.0);

  const torch::Tensor log_probs = torch::log_softmax(value_logits, -1);
  const torch::Tensor lower_log_probs = log_probs.gather(-1, lower.unsqueeze(-1)).squeeze(-1);
  const torch::Tensor upper_log_probs = log_probs.gather(-1, upper.unsqueeze(-1)).squeeze(-1);

  const torch::Tensor projection =
      lower_log_probs * (1.0 - weight_upper) + upper_log_probs * weight_upper;
  return -projection.mean();
}

torch::Tensor sample_quantile_value(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support) {
  torch::Tensor sampled_indices = torch::multinomial(
      torch::softmax(value_logits, -1), 1, true).squeeze(-1);
  return atom_support.index_select(0, sampled_indices.view({-1}))
      .view_as(sampled_indices);
}

torch::Tensor compute_mean_value(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "compute_mean_value");
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  return (probs * atom_support).sum(-1);
}

torch::Tensor compute_distribution_variance(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "distribution_variance");
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  const torch::Tensor expected = (probs * atom_support).sum(-1, true);
  const torch::Tensor shifted_support = atom_support.unsqueeze(0) - expected;
  const torch::Tensor variance = (probs * shifted_support.pow(2)).sum(-1);
  return variance.clamp_min(1.0e-6F);
}

torch::Tensor compute_distribution_entropy(
    const torch::Tensor& value_logits) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "distribution_entropy");
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  const float eps = 1.0e-8F;
  return -(probs * torch::log(probs + eps)).sum(-1);
}

float compute_adaptive_epsilon(
    const torch::Tensor& variance,
    float epsilon_base,
    float epsilon_beta,
    float epsilon_min,
    float epsilon_max) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "adaptive_epsilon");
  const float mean_variance = variance.mean().item<float>();
  float adaptive = epsilon_base / (1.0F + epsilon_beta * mean_variance);
  return std::clamp(adaptive, epsilon_min, epsilon_max);
}

torch::Tensor compute_adaptive_epsilon_tensor(
    const torch::Tensor& variance,
    float epsilon_base,
    float epsilon_beta,
    float epsilon_min,
    float epsilon_max) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "adaptive_epsilon_tensor");
  torch::Tensor adaptive = epsilon_base / (1.0F + epsilon_beta * variance);
  return adaptive.clamp(epsilon_min, epsilon_max);
}

torch::Tensor compute_confidence_weights(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support,
    const std::string& weight_type,
    float weight_delta,
    bool normalize) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "confidence_weights");
  torch::Tensor raw_weights;
  if (weight_type == "entropy") {
    const torch::Tensor entropy = compute_distribution_entropy(value_logits);
    raw_weights = 1.0 / (entropy + weight_delta);
  } else if (weight_type == "variance") {
    const torch::Tensor variance = compute_distribution_variance(value_logits, atom_support);
    raw_weights = 1.0 / (variance + weight_delta);
  } else {
    return torch::ones({value_logits.size(0)}, value_logits.options());
  }
  if (normalize) {
    raw_weights = raw_weights / (raw_weights.mean().clamp_min(1.0e-8F));
  }
  return raw_weights.detach();
}

torch::Tensor normalize_advantage(const torch::Tensor& advantages, const torch::Tensor& active_mask) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "normalize_advantage");
  const int64_t active_count = active_mask.sum().item<int64_t>();
  if (active_count <= 0) {
    return advantages;
  }
  const torch::Tensor active_adv = advantages.masked_select(active_mask > 0.5F);
  const torch::Tensor mean = active_adv.mean();
  if (active_count <= 1) {
    return advantages - mean;
  }
  const torch::Tensor std = active_adv.std(false).clamp_min(1.0e-8);
  return (advantages - mean) / std;
}

torch::Tensor sample_future_goal_positions(
    const torch::Tensor& goal_positions,
    const torch::Tensor& dones,
    const torch::Tensor& episode_starts,
    int max_future) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "sample_future_goal_positions");
  const int64_t steps = goal_positions.size(0);
  const int64_t agents = goal_positions.size(1);
  const int64_t dim = goal_positions.size(2);
  torch::Tensor goal_cpu = goal_positions.to(torch::kCPU).to(torch::kFloat32).contiguous();
  torch::Tensor dones_cpu = dones.defined()
      ? dones.to(torch::kCPU).to(torch::kFloat32).contiguous()
      : torch::Tensor{};
  torch::Tensor starts_cpu = episode_starts.defined()
      ? episode_starts.to(torch::kCPU).to(torch::kFloat32).contiguous()
      : torch::Tensor{};
  torch::Tensor sampled_cpu = torch::empty_like(goal_cpu);

  const int64_t horizon = std::max<int64_t>(1, max_future);
  const float* dones_ptr = dones_cpu.defined() ? dones_cpu.data_ptr<float>() : nullptr;
  const float* starts_ptr = starts_cpu.defined() ? starts_cpu.data_ptr<float>() : nullptr;

  for (int64_t t = 0; t < steps; ++t) {
    for (int64_t a = 0; a < agents; ++a) {
      int64_t end_idx = steps;
      for (int64_t j = t; j < steps; ++j) {
        const int64_t ja = j * agents + a;
        if (dones_ptr != nullptr && dones_ptr[ja] > 0.5F) {
          end_idx = j + 1;
          break;
        }
        if (starts_ptr != nullptr && starts_ptr[ja] > 0.5F && j > t) {
          end_idx = j;
          break;
        }
      }

      const int64_t max_offset = std::min<int64_t>(end_idx - t - 1, horizon);
      if (max_offset <= 0) {
        for (int64_t d = 0; d < dim; ++d) {
          sampled_cpu[t][a][d] = goal_cpu[t][a][d];
        }
        continue;
      }

      const int64_t chosen_offset = std::rand() % max_offset;
      const int64_t chosen = t + 1 + chosen_offset;
      for (int64_t d = 0; d < dim; ++d) {
        sampled_cpu[t][a][d] = goal_cpu[chosen][a][d];
      }
    }
  }

  torch::Tensor sampled = sampled_cpu;
  if (goal_positions.scalar_type() != torch::kFloat32) {
    sampled = sampled.to(goal_positions.scalar_type());
  }
  if (!goal_positions.device().is_cpu()) {
    sampled = sampled.to(goal_positions.device());
  }
  return sampled;
}

torch::Tensor compute_pairwise_negative_l2_logits(
    const torch::Tensor& lhs_embeddings,
    const torch::Tensor& rhs_embeddings) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "infonce_logits");
  const torch::Tensor diff = lhs_embeddings.unsqueeze(1) - rhs_embeddings.unsqueeze(0);
  return -(diff.square().sum(-1).clamp_min(1.0e-8F));
}

torch::Tensor compute_symmetric_infonce_loss(
    const torch::Tensor& logits,
    float logsumexp_penalty_coeff) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "infonce_loss");
  const torch::Tensor diag = logits.diagonal();
  const torch::Tensor row_lse = torch::logsumexp(logits, 1);
  const torch::Tensor col_lse = torch::logsumexp(logits, 0);
  const torch::Tensor row_loss = -(diag - row_lse).mean();
  const torch::Tensor col_loss = -(diag - col_lse).mean();
  const torch::Tensor penalty = logsumexp_penalty_coeff * (row_lse.mean() + col_lse.mean());
  return row_loss + col_loss + penalty;
}

float compute_discrete_policy_kl(
    const torch::Tensor& base_logits,
    const torch::Tensor& perturbed_logits,
    const torch::Tensor& action_masks) {
  const torch::Tensor base_masked = apply_action_mask_to_logits(base_logits, action_masks);
  const torch::Tensor perturbed_masked = apply_action_mask_to_logits(perturbed_logits, action_masks);

  const torch::Tensor base_probs = torch::softmax(base_masked, -1);
  const torch::Tensor perturbed_probs = torch::softmax(perturbed_masked, -1);

  const torch::Tensor kl = (perturbed_probs * (torch::log(perturbed_probs + 1.0e-8) - torch::log(base_probs + 1.0e-8))).sum(-1);
  return kl.mean().item<float>();
}

ContinuumState detach_state(ContinuumState state) {
  state.workspace = detach_tensor(state.workspace);
  state.stm_keys = detach_tensor(state.stm_keys);
  state.stm_values = detach_tensor(state.stm_values);
  state.stm_strengths = detach_tensor(state.stm_strengths);
  state.stm_write_index = detach_tensor(state.stm_write_index);
  state.ltm_coeffs = detach_tensor(state.ltm_coeffs);
  state.timestep = detach_tensor(state.timestep);
  return state;
}

ContinuumState clone_state(const ContinuumState& state) {
  return {
      clone_tensor(state.workspace),
      clone_tensor(state.stm_keys),
      clone_tensor(state.stm_values),
      clone_tensor(state.stm_strengths),
      clone_tensor(state.stm_write_index),
      clone_tensor(state.ltm_coeffs),
      clone_tensor(state.timestep),
  };
}

ContinuumState state_to_device(ContinuumState state, const torch::Device& device) {
  state.workspace = tensor_to_device(state.workspace, device);
  state.stm_keys = tensor_to_device(state.stm_keys, device);
  state.stm_values = tensor_to_device(state.stm_values, device);
  state.stm_strengths = tensor_to_device(state.stm_strengths, device);
  state.stm_write_index = tensor_to_device(state.stm_write_index, device);
  state.ltm_coeffs = tensor_to_device(state.ltm_coeffs, device);
  state.timestep = tensor_to_device(state.timestep, device);
  return state;
}

ContinuumState gather_state(const ContinuumState& state, const torch::Tensor& indices) {
  return {
      gather_tensor(state.workspace, indices),
      gather_tensor(state.stm_keys, indices),
      gather_tensor(state.stm_values, indices),
      gather_tensor(state.stm_strengths, indices),
      gather_tensor(state.stm_write_index, indices),
      gather_tensor(state.ltm_coeffs, indices),
      gather_tensor(state.timestep, indices),
  };
}

void scatter_state(ContinuumState& dst, const torch::Tensor& indices, const ContinuumState& src) {
  scatter_tensor(dst.workspace, indices, src.workspace);
  scatter_tensor(dst.stm_keys, indices, src.stm_keys);
  scatter_tensor(dst.stm_values, indices, src.stm_values);
  scatter_tensor(dst.stm_strengths, indices, src.stm_strengths);
  scatter_tensor(dst.stm_write_index, indices, src.stm_write_index);
  scatter_tensor(dst.ltm_coeffs, indices, src.ltm_coeffs);
  scatter_tensor(dst.timestep, indices, src.timestep);
}

}  // namespace pulsar

#endif
