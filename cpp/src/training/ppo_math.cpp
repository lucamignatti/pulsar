#include "pulsar/training/ppo_math.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {
namespace {

torch::Tensor sample_categorical_from_logits(const torch::Tensor& logits) {
  const torch::Tensor uniform = torch::rand_like(logits).clamp_(1.0e-6, 1.0 - 1.0e-6);
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
  return logits.masked_fill(action_masks.logical_not(), -1.0e9);
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
  const torch::Tensor masked = apply_action_mask_to_logits(logits, action_masks);
  const torch::Tensor probs = torch::softmax(masked, -1);
  const torch::Tensor valid_counts = action_masks.to(torch::kFloat32).sum(-1).clamp_min(1.0F);
  return -(probs * torch::log(probs + 1.0e-8)).sum(-1) / valid_counts.log().clamp_min(1.0e-6);
}

torch::Tensor compute_gae(
    const torch::Tensor& values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda) {
  const int64_t steps = values.size(0);
  const int64_t agents = values.size(1);
  torch::Tensor advantages = torch::zeros({steps, agents}, values.options());
  torch::Tensor last_gae = torch::zeros({agents}, values.options());

  for (int64_t t = steps - 1; t >= 0; --t) {
    const torch::Tensor next_value = (t < steps - 1) ? values[t + 1] : torch::zeros({agents}, values.options());
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
  const torch::Tensor ratio = torch::exp(current_log_probs - old_log_probs);
  const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - clip_range, 1.0 + clip_range);
  return -torch::min(ratio * advantages, clipped_ratio * advantages).mean();
}

torch::Tensor distributional_value_loss(
    const torch::Tensor& value_logits,
    const torch::Tensor& returns,
    const torch::Tensor& atom_support,
    float v_min,
    float v_max,
    int num_atoms) {
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
  const torch::Tensor dist = torch::distributions::Categorical{}.sample(
      [&](const torch::Tensor& logits) { return logits; }(value_logits));
  torch::Tensor sampled_indices = torch::multinomial(
      torch::softmax(value_logits, -1), 1, true).squeeze(-1);
  return atom_support.index_select(0, sampled_indices.view({-1}))
      .view_as(sampled_indices);
}

torch::Tensor compute_mean_value(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support) {
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  return (probs * atom_support).sum(-1);
}

torch::Tensor compute_distribution_variance(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support) {
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  const torch::Tensor expected = (probs * atom_support).sum(-1);
  const torch::Tensor expected_sq = (probs * atom_support.pow(2)).sum(-1);
  return torch::relu(expected_sq - expected.pow(2));
}

torch::Tensor compute_distribution_entropy(
    const torch::Tensor& value_logits) {
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
  const float mean_variance = variance.mean().item<float>();
  float adaptive = epsilon_base / (1.0F + epsilon_beta * mean_variance);
  return std::clamp(adaptive, epsilon_min, epsilon_max);
}

torch::Tensor compute_confidence_weights(
    const torch::Tensor& value_logits,
    const torch::Tensor& atom_support,
    const std::string& weight_type,
    float weight_delta,
    bool normalize) {
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
    raw_weights = raw_weights / (raw_weights.mean() + 1.0e-8F);
  }
  return raw_weights.detach();
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
