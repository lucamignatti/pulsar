#include "pulsar/training/lfpo_math.hpp"

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

torch::Tensor latent_action_scores(const torch::Tensor& outcome_logits) {
  const torch::Tensor log_probs = torch::log_softmax(outcome_logits, -1);
  const torch::Tensor score_logp = log_probs.select(-1, 0);
  const torch::Tensor concede_logp = log_probs.select(-1, 1);
  return (score_logp - concede_logp).mean(-1);
}

torch::Tensor relative_candidate_advantages(const torch::Tensor& candidate_scores) {
  torch::Tensor advantages = candidate_scores - candidate_scores.mean(-1, true);
  const torch::Tensor std = advantages.pow(2).mean(-1, true).sqrt().clamp_min(1.0e-6);
  return (advantages / std).detach();
}

torch::Tensor clipped_lfpo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range) {
  const torch::Tensor ratio = torch::exp(current_log_probs - old_log_probs);
  const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - clip_range, 1.0 + clip_range);
  return -torch::min(ratio * advantages, clipped_ratio * advantages).mean();
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
