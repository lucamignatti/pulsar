#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "pulsar/training/ppo_math.hpp"
#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

void seed_everything(std::uint64_t seed) {
  torch::manual_seed(static_cast<int64_t>(seed));
  std::srand(static_cast<unsigned int>(seed));
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
  torch::Tensor masked = apply_action_mask_to_logits(logits, action_masks);
  {
    const torch::Tensor all_invalid = action_masks.to(torch::kFloat32).sum(-1) <= 0.5F;
    if (all_invalid.any().item<bool>()) {
      throw std::invalid_argument(
          "sample_masked_actions: all actions invalid for at least one agent. "
          "The action mask builder must guarantee at least one valid noop action.");
    }
  }
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
    const torch::Tensor& next_values,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_values) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "compute_gae");
  const int64_t steps = values.size(0);
  const int64_t agents = values.size(1);
  torch::Tensor advantages = torch::zeros({steps, agents}, values.options());
  torch::Tensor last_gae = torch::zeros({agents}, values.options());

  const torch::Tensor boundary_value = next_values.defined()
      ? next_values.to(values.device()).to(values.dtype())
      : torch::zeros({agents}, values.options());

  for (int64_t t = steps - 1; t >= 0; --t) {
    torch::Tensor next_value = (t < steps - 1) ? values[t + 1] : boundary_value;
    if (bootstrap_values.defined()) {
      next_value = torch::where(bootstrap_mask[t] > 0.5F, bootstrap_values[t], next_value);
    }
    const torch::Tensor non_terminal = 1.0F - dones[t];
    const torch::Tensor delta_mult = bootstrap_mask.defined()
        ? torch::where(bootstrap_mask[t] > 0.5F, torch::ones_like(dones[t]), non_terminal)
        : non_terminal;
    const torch::Tensor delta = rewards[t] + gamma * next_value * delta_mult - values[t];
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
  constexpr float kMaxSquaredDiff = 1.0e6F;
  const torch::Tensor lhs = lhs_embeddings.to(torch::kFloat32);
  const torch::Tensor rhs = rhs_embeddings.to(torch::kFloat32);
  const torch::Tensor diff = lhs.unsqueeze(1) - rhs.unsqueeze(0);
  const torch::Tensor squared = diff.square().clamp_max(kMaxSquaredDiff);
  return -(squared.sum(-1).clamp_min(1.0e-8F));
}

torch::Tensor compute_symmetric_infonce_loss(
    const torch::Tensor& logits,
    float logsumexp_penalty_coeff) {
  PULSAR_TRACE_SCOPE_CAT("ppo_math", "infonce_loss");
  constexpr float kFiniteLogitClamp = 1.0e6F;
  const torch::Tensor logits_f32 = logits.to(torch::kFloat32).clamp(-kFiniteLogitClamp, kFiniteLogitClamp);
  const torch::Tensor diag = logits_f32.diagonal();
  const torch::Tensor row_lse = torch::logsumexp(logits_f32, 1);
  const torch::Tensor col_lse = torch::logsumexp(logits_f32, 0);
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

}  // namespace pulsar

#endif
