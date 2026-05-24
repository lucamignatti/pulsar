#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsar/training/vrpo_math.hpp"
#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

namespace {

constexpr float kAdvantageStdFloor = 1.0F;
constexpr float kAdvantageAbsCap = 10.0F;

torch::Tensor clamp_normalized_advantage(const torch::Tensor& normalized) {
  return normalized.clamp(-kAdvantageAbsCap, kAdvantageAbsCap);
}

std::uint32_t get_thread_local_seed() {
  thread_local std::random_device rd;
  thread_local std::mt19937 generator(rd());
  return generator();
}

int64_t get_thread_local_random(int64_t max_val) {
  if (max_val <= 0) return 0;
  thread_local std::random_device rd;
  thread_local std::mt19937 generator(rd());
  std::uniform_int_distribution<int64_t> dist(0, max_val - 1);
  return dist(generator);
}

}  // namespace

#ifdef PULSAR_HAS_PPO_MATH_CUDA_KERNELS
// Keep raw CUDA kernel signatures untouched for compiling safety
torch::Tensor compute_gae_cuda(
    const torch::Tensor& values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_values,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_values);

torch::Tensor compute_q_boosted_gae_cuda(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_v_from_q);

torch::Tensor compute_centered_expected_sarsa_gae_cuda(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_v_from_q);

torch::Tensor normalize_advantage_cuda(const torch::Tensor& advantages, const torch::Tensor& active_mask);

std::vector<torch::Tensor> sample_masked_actions_cuda(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    bool need_log_probs,
    std::uint32_t seed,
    float temperature);

torch::Tensor masked_action_entropy_cuda(const torch::Tensor& logits, const torch::Tensor& action_masks, float temperature);

torch::Tensor clipped_ppo_policy_loss_forward_cuda(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range);

torch::Tensor clipped_ppo_policy_loss_backward_cuda(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    const torch::Tensor& grad_output,
    float clip_range);

torch::Tensor sample_future_goal_positions_cuda(
    const torch::Tensor& goal_positions,
    const torch::Tensor& dones,
    const torch::Tensor& episode_starts,
    int max_future,
    std::uint32_t seed);

torch::Tensor compute_sparse_event_soon_targets_cuda(
    const torch::Tensor& sparse_events,
    const torch::Tensor& dones,
    const torch::Tensor& horizons);
#endif

#ifdef PULSAR_HAS_PPO_MATH_HIP_KERNELS
// Keep raw HIP kernel signatures untouched for compiling safety
torch::Tensor compute_gae_hip(
    const torch::Tensor& values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_values,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_values);

torch::Tensor compute_q_boosted_gae_hip(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_v_from_q);

torch::Tensor compute_centered_expected_sarsa_gae_hip(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_v_from_q);

torch::Tensor normalize_advantage_hip(const torch::Tensor& advantages, const torch::Tensor& active_mask);

std::vector<torch::Tensor> sample_masked_actions_hip(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    bool need_log_probs,
    std::uint32_t seed,
    float temperature);

torch::Tensor masked_action_entropy_hip(const torch::Tensor& logits, const torch::Tensor& action_masks, float temperature);

torch::Tensor clipped_ppo_policy_loss_forward_hip(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range);

torch::Tensor clipped_ppo_policy_loss_backward_hip(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    const torch::Tensor& grad_output,
    float clip_range);

torch::Tensor sample_future_goal_positions_hip(
    const torch::Tensor& goal_positions,
    const torch::Tensor& dones,
    const torch::Tensor& episode_starts,
    int max_future,
    std::uint32_t seed);

torch::Tensor compute_sparse_event_soon_targets_hip(
    const torch::Tensor& sparse_events,
    const torch::Tensor& dones,
    const torch::Tensor& horizons);
#endif

void seed_everything(std::uint64_t seed) {
  torch::manual_seed(static_cast<int64_t>(seed));
  std::srand(static_cast<unsigned int>(seed));
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed_all(static_cast<int64_t>(seed));
  }
}

namespace {

constexpr float kMaskedLogit = -1.0e9F;

torch::Tensor sample_categorical_from_logits(const torch::Tensor& logits) {
  const torch::Tensor uniform = torch::rand_like(logits).clamp_min_(1.0e-6);
  const torch::Tensor gumbel = -torch::log(-torch::log(uniform));
  return (logits + gumbel).argmax(-1);
}

bool can_use_ppo_math_accel(const torch::Tensor& tensor) {
#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS) || defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
  return tensor.defined() && tensor.is_cuda() &&
      (tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat16);
#else
  (void)tensor;
  return false;
#endif
}

bool env_flag_disabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr &&
      (std::strcmp(value, "0") == 0 ||
       std::strcmp(value, "false") == 0 ||
       std::strcmp(value, "FALSE") == 0 ||
       std::strcmp(value, "no") == 0);
}

bool can_use_action_accel(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  return can_use_ppo_math_accel(logits) &&
      action_masks.defined() &&
      action_masks.is_cuda() &&
      action_masks.dim() == logits.dim() &&
      action_masks.size(-1) == logits.size(-1) &&
      (action_masks.scalar_type() == torch::kBool || action_masks.scalar_type() == torch::kUInt8);
}

}  // namespace

torch::Tensor compute_sparse_event_soon_targets(
    const torch::Tensor& sparse_events,
    const torch::Tensor& dones,
    const torch::Tensor& horizons) {
  if (!sparse_events.defined() || sparse_events.dim() != 3) {
    throw std::invalid_argument("compute_sparse_event_soon_targets expects [T,N,C] sparse_events.");
  }
  if (!dones.defined() || dones.dim() != 2 ||
      dones.size(0) != sparse_events.size(0) ||
      dones.size(1) != sparse_events.size(1)) {
    throw std::invalid_argument("compute_sparse_event_soon_targets dones shape must be [T,N].");
  }
  if (!horizons.defined() || horizons.dim() != 1 ||
      horizons.size(0) != sparse_events.size(2)) {
    throw std::invalid_argument("compute_sparse_event_soon_targets horizons shape must be [C].");
  }
#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS)
  if (sparse_events.is_cuda()) {
    return compute_sparse_event_soon_targets_cuda(
        sparse_events.contiguous(),
        dones.contiguous(),
        horizons.to(torch::TensorOptions().device(sparse_events.device()).dtype(torch::kInt32)).contiguous());
  }
#elif defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
  if (sparse_events.is_cuda()) {
    return compute_sparse_event_soon_targets_hip(
        sparse_events.contiguous(),
        dones.contiguous(),
        horizons.to(torch::TensorOptions().device(sparse_events.device()).dtype(torch::kInt32)).contiguous());
  }
#endif
  const torch::Tensor events_cpu = sparse_events.to(torch::kCPU).contiguous();
  const torch::Tensor dones_cpu = dones.to(torch::kCPU).contiguous();
  const torch::Tensor horizons_cpu =
      horizons.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32)).contiguous();
  const int T = static_cast<int>(events_cpu.size(0));
  const int N = static_cast<int>(events_cpu.size(1));
  const int C = static_cast<int>(events_cpu.size(2));
  torch::Tensor targets = torch::zeros({T, N, C}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  const auto* events_ptr = events_cpu.data_ptr<std::uint8_t>();
  const auto* dones_ptr = dones_cpu.data_ptr<float>();
  const auto* horizons_ptr = horizons_cpu.data_ptr<int>();
  auto* targets_ptr = targets.data_ptr<std::int64_t>();
  for (int t = 0; t < T; ++t) {
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        const int horizon = std::max(1, horizons_ptr[c]);
        for (int k = 0; k < horizon; ++k) {
          const int look = t + k;
          if (look >= T) break;
          const int event_index = (look * N + n) * C + c;
          if (events_ptr[event_index] != 0) {
            targets_ptr[(t * N + n) * C + c] = 1;
            break;
          }
          if (dones_ptr[look * N + n] > 0.5F) {
            break;
          }
        }
      }
    }
  }
  return targets.to(sparse_events.device());
}

torch::Tensor apply_action_mask_to_logits(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  return logits.masked_fill(action_masks.logical_not(), kMaskedLogit);
}

torch::Tensor sample_masked_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs,
    float temperature) {
  const float safe_temperature = std::max(temperature, 1.0e-6F);
  if (can_use_action_accel(logits, action_masks) && logits.dim() == 2) {
    const bool need_log_probs = log_probs != nullptr;
    const std::uint32_t seed = get_thread_local_seed();
#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS)
    std::vector<torch::Tensor> result = sample_masked_actions_cuda(
        logits.contiguous(),
        action_masks.contiguous(),
        deterministic,
        need_log_probs,
        seed,
        safe_temperature);
    if (need_log_probs) {
      *log_probs = result[1];
    }
    return result[0];
#elif defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
    std::vector<torch::Tensor> result = sample_masked_actions_hip(
        logits.contiguous(),
        action_masks.contiguous(),
        deterministic,
        need_log_probs,
        seed,
        safe_temperature);
    if (need_log_probs) {
      *log_probs = result[1];
    }
    return result[0];
#endif
  }
  torch::Tensor masked = apply_action_mask_to_logits(logits / safe_temperature, action_masks);
  const torch::Tensor actions = deterministic ? masked.argmax(-1) : sample_categorical_from_logits(masked);
  if (log_probs != nullptr) {
    *log_probs = torch::log_softmax(masked, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
  }
  return actions;
}

torch::Tensor masked_action_entropy(const torch::Tensor& logits, const torch::Tensor& action_masks, float temperature) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "entropy");
  const float safe_temperature = std::max(temperature, 1.0e-6F);
  if (!logits.requires_grad() && can_use_action_accel(logits, action_masks) && logits.dim() == 2) {
#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS)
    return masked_action_entropy_cuda(logits.contiguous(), action_masks.contiguous(), safe_temperature);
#elif defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
    return masked_action_entropy_hip(logits.contiguous(), action_masks.contiguous(), safe_temperature);
#endif
  }
  const torch::Tensor masked = apply_action_mask_to_logits(logits / safe_temperature, action_masks);
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
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "compute_gae");
  if (can_use_ppo_math_accel(values) &&
      rewards.is_cuda() &&
      dones.is_cuda() &&
      rewards.scalar_type() == torch::kFloat32 &&
      dones.scalar_type() == torch::kFloat32 &&
      (!next_values.defined() || (next_values.is_cuda() && next_values.scalar_type() == torch::kFloat32)) &&
      (!bootstrap_mask.defined() || (bootstrap_mask.is_cuda() && bootstrap_mask.scalar_type() == torch::kFloat32)) &&
      (!bootstrap_values.defined() || (bootstrap_values.is_cuda() && bootstrap_values.scalar_type() == torch::kFloat32))) {
#ifdef PULSAR_HAS_PPO_MATH_CUDA_KERNELS
    return compute_gae_cuda(
        values.contiguous(),
        rewards.contiguous(),
        dones.contiguous(),
        gamma,
        gae_lambda,
        next_values.defined() ? next_values.contiguous() : torch::Tensor{},
        bootstrap_mask.defined() ? bootstrap_mask.contiguous() : torch::Tensor{},
        bootstrap_values.defined() ? bootstrap_values.contiguous() : torch::Tensor{});
#endif
#ifdef PULSAR_HAS_PPO_MATH_HIP_KERNELS
    return compute_gae_hip(
        values.contiguous(),
        rewards.contiguous(),
        dones.contiguous(),
        gamma,
        gae_lambda,
        next_values.defined() ? next_values.contiguous() : torch::Tensor{},
        bootstrap_mask.defined() ? bootstrap_mask.contiguous() : torch::Tensor{},
        bootstrap_values.defined() ? bootstrap_values.contiguous() : torch::Tensor{});
#endif
  }
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

torch::Tensor compute_q_boosted_gae(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_v_from_q) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "compute_q_boosted_gae");
  if (can_use_ppo_math_accel(q_values_taken) &&
      v_from_q.is_cuda() &&
      rewards.is_cuda() &&
      dones.is_cuda() &&
      v_from_q.scalar_type() == torch::kFloat32 &&
      rewards.scalar_type() == torch::kFloat32 &&
      dones.scalar_type() == torch::kFloat32 &&
      (!next_v_from_q.defined() || (next_v_from_q.is_cuda() && next_v_from_q.scalar_type() == torch::kFloat32)) &&
      (!bootstrap_mask.defined() || (bootstrap_mask.is_cuda() && bootstrap_mask.scalar_type() == torch::kFloat32)) &&
      (!bootstrap_v_from_q.defined() || (bootstrap_v_from_q.is_cuda() && bootstrap_v_from_q.scalar_type() == torch::kFloat32))) {
#ifdef PULSAR_HAS_PPO_MATH_CUDA_KERNELS
    return compute_q_boosted_gae_cuda(
        q_values_taken.contiguous(),
        v_from_q.contiguous(),
        rewards.contiguous(),
        dones.contiguous(),
        gamma,
        gae_lambda,
        next_v_from_q.defined() ? next_v_from_q.contiguous() : torch::Tensor{},
        bootstrap_mask.defined() ? bootstrap_mask.contiguous() : torch::Tensor{},
        bootstrap_v_from_q.defined() ? bootstrap_v_from_q.contiguous() : torch::Tensor{});
#endif
#ifdef PULSAR_HAS_PPO_MATH_HIP_KERNELS
    return compute_q_boosted_gae_hip(
        q_values_taken.contiguous(),
        v_from_q.contiguous(),
        rewards.contiguous(),
        dones.contiguous(),
        gamma,
        gae_lambda,
        next_v_from_q.defined() ? next_v_from_q.contiguous() : torch::Tensor{},
        bootstrap_mask.defined() ? bootstrap_mask.contiguous() : torch::Tensor{},
        bootstrap_v_from_q.defined() ? bootstrap_v_from_q.contiguous() : torch::Tensor{});
#endif
  }
  const int64_t steps = q_values_taken.size(0);
  const int64_t agents = q_values_taken.size(1);
  torch::Tensor advantages = torch::zeros({steps, agents}, q_values_taken.options());
  torch::Tensor last_gae = torch::zeros({agents}, q_values_taken.options());

  const torch::Tensor boundary_v_from_q = next_v_from_q.defined()
      ? next_v_from_q.to(q_values_taken.device()).to(q_values_taken.dtype())
      : torch::zeros({agents}, q_values_taken.options());

  for (int64_t t = steps - 1; t >= 0; --t) {
    torch::Tensor next_v = (t < steps - 1) ? v_from_q[t + 1] : boundary_v_from_q;
    if (bootstrap_v_from_q.defined()) {
      next_v = torch::where(bootstrap_mask[t] > 0.5F, bootstrap_v_from_q[t], next_v);
    }
    const torch::Tensor non_terminal = 1.0F - dones[t];
    const torch::Tensor delta_mult = bootstrap_mask.defined()
        ? torch::where(bootstrap_mask[t] > 0.5F, torch::ones_like(dones[t]), non_terminal)
        : non_terminal;
    const torch::Tensor delta = rewards[t] + gamma * next_v * delta_mult - q_values_taken[t];
    last_gae = delta + gamma * gae_lambda * non_terminal * last_gae;
    advantages[t] = q_values_taken[t] - v_from_q[t] + last_gae;
  }
  return advantages;
}

torch::Tensor compute_centered_expected_sarsa_gae(
    const torch::Tensor& q_values_taken,
    const torch::Tensor& v_from_q,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda,
    const torch::Tensor& next_v_from_q,
    const torch::Tensor& bootstrap_mask,
    const torch::Tensor& bootstrap_v_from_q) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "compute_centered_expected_sarsa_gae");
  if (can_use_ppo_math_accel(q_values_taken) &&
      v_from_q.is_cuda() &&
      rewards.is_cuda() &&
      dones.is_cuda() &&
      v_from_q.scalar_type() == torch::kFloat32 &&
      rewards.scalar_type() == torch::kFloat32 &&
      dones.scalar_type() == torch::kFloat32 &&
      (!next_v_from_q.defined() || (next_v_from_q.is_cuda() && next_v_from_q.scalar_type() == torch::kFloat32)) &&
      (!bootstrap_mask.defined() || (bootstrap_mask.is_cuda() && bootstrap_mask.scalar_type() == torch::kFloat32)) &&
      (!bootstrap_v_from_q.defined() || (bootstrap_v_from_q.is_cuda() && bootstrap_v_from_q.scalar_type() == torch::kFloat32))) {
#ifdef PULSAR_HAS_PPO_MATH_CUDA_KERNELS
    return compute_centered_expected_sarsa_gae_cuda(
        q_values_taken.contiguous(),
        v_from_q.contiguous(),
        rewards.contiguous(),
        dones.contiguous(),
        gamma,
        gae_lambda,
        next_v_from_q.defined() ? next_v_from_q.contiguous() : torch::Tensor{},
        bootstrap_mask.defined() ? bootstrap_mask.contiguous() : torch::Tensor{},
        bootstrap_v_from_q.defined() ? bootstrap_v_from_q.contiguous() : torch::Tensor{});
#endif
#ifdef PULSAR_HAS_PPO_MATH_HIP_KERNELS
    return compute_centered_expected_sarsa_gae_hip(
        q_values_taken.contiguous(),
        v_from_q.contiguous(),
        rewards.contiguous(),
        dones.contiguous(),
        gamma,
        gae_lambda,
        next_v_from_q.defined() ? next_v_from_q.contiguous() : torch::Tensor{},
        bootstrap_mask.defined() ? bootstrap_mask.contiguous() : torch::Tensor{},
        bootstrap_v_from_q.defined() ? bootstrap_v_from_q.contiguous() : torch::Tensor{});
#endif
  }
  const int64_t steps = q_values_taken.size(0);
  const int64_t agents = q_values_taken.size(1);
  torch::Tensor advantages = torch::zeros({steps, agents}, q_values_taken.options());
  torch::Tensor last_gae = torch::zeros({agents}, q_values_taken.options());

  const torch::Tensor boundary_v_from_q = next_v_from_q.defined()
      ? next_v_from_q.to(q_values_taken.device()).to(q_values_taken.dtype())
      : torch::zeros({agents}, q_values_taken.options());

  for (int64_t t = steps - 1; t >= 0; --t) {
    torch::Tensor next_v = (t < steps - 1) ? v_from_q[t + 1] : boundary_v_from_q;
    if (bootstrap_v_from_q.defined()) {
      next_v = torch::where(bootstrap_mask[t] > 0.5F, bootstrap_v_from_q[t], next_v);
    }
    const torch::Tensor non_terminal = 1.0F - dones[t];
    const torch::Tensor delta_mult = bootstrap_mask.defined()
        ? torch::where(bootstrap_mask[t] > 0.5F, torch::ones_like(dones[t]), non_terminal)
        : non_terminal;
    // Centered residual: subtract V(s) instead of Q(s, a_t).
    const torch::Tensor delta = rewards[t] + gamma * next_v * delta_mult - v_from_q[t];
    last_gae = delta + gamma * gae_lambda * non_terminal * last_gae;
    advantages[t] = q_values_taken[t] - v_from_q[t] + last_gae;
  }
  return advantages;
}

namespace {

#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS) || defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
class ClippedVrpoPolicyLossAccelFunction
    : public torch::autograd::Function<ClippedVrpoPolicyLossAccelFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor current_log_probs,
      torch::Tensor old_log_probs,
      torch::Tensor advantages,
      double clip_range) {
    ctx->save_for_backward({current_log_probs, old_log_probs, advantages});
    ctx->saved_data["clip_range"] = clip_range;
#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS)
    return clipped_ppo_policy_loss_forward_cuda(
        current_log_probs.contiguous(),
        old_log_probs.contiguous(),
        advantages.contiguous(),
        static_cast<float>(clip_range));
#elif defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
    return clipped_ppo_policy_loss_forward_hip(
        current_log_probs.contiguous(),
        old_log_probs.contiguous(),
        advantages.contiguous(),
        static_cast<float>(clip_range));
#endif
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const float clip_range = static_cast<float>(ctx->saved_data["clip_range"].toDouble());
    const torch::Tensor grad_output = grad_outputs[0].contiguous();
#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS)
    torch::Tensor grad_current = clipped_ppo_policy_loss_backward_cuda(
        saved[0].contiguous(),
        saved[1].contiguous(),
        saved[2].contiguous(),
        grad_output,
        clip_range);
#elif defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
    torch::Tensor grad_current = clipped_ppo_policy_loss_backward_hip(
        saved[0].contiguous(),
        saved[1].contiguous(),
        saved[2].contiguous(),
        grad_output,
        clip_range);
#endif
    return {grad_current, torch::Tensor{}, torch::Tensor{}, torch::Tensor{}};
  }
};
#endif

bool can_use_clipped_loss_accel(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages) {
  return !env_flag_disabled("PULSAR_PPO_LOSS_ACCEL") &&
      can_use_ppo_math_accel(current_log_probs) &&
      old_log_probs.is_cuda() &&
      advantages.is_cuda() &&
      old_log_probs.scalar_type() == torch::kFloat32 &&
      advantages.scalar_type() == torch::kFloat32 &&
      current_log_probs.sizes() == old_log_probs.sizes() &&
      current_log_probs.sizes() == advantages.sizes();
}

}  // namespace

torch::Tensor clipped_vrpo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    float clip_range) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "clipped_vrpo_loss");
  if (can_use_clipped_loss_accel(current_log_probs, old_log_probs, advantages)) {
#if defined(PULSAR_HAS_PPO_MATH_CUDA_KERNELS) || defined(PULSAR_HAS_PPO_MATH_HIP_KERNELS)
    return ClippedVrpoPolicyLossAccelFunction::apply(
        current_log_probs,
        old_log_probs,
        advantages,
        static_cast<double>(clip_range));
#endif
  }
  const torch::Tensor ratio = torch::exp(current_log_probs - old_log_probs);
  const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - clip_range, 1.0 + clip_range);
  return -torch::min(ratio * advantages, clipped_ratio * advantages);
}

torch::Tensor clipped_vrpo_policy_loss(
    const torch::Tensor& current_log_probs,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    const torch::Tensor& clip_range) {
  const torch::Tensor ratio = torch::exp(current_log_probs - old_log_probs);
  const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - clip_range, 1.0 + clip_range);
  return -torch::min(ratio * advantages, clipped_ratio * advantages);
}

torch::Tensor normalize_advantage(const torch::Tensor& advantages, const torch::Tensor& active_mask) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "normalize_advantage");
  if (can_use_ppo_math_accel(advantages) && active_mask.defined() && active_mask.is_cuda()) {
    const torch::Tensor active_arg = active_mask.scalar_type() == torch::kFloat32
        ? active_mask.contiguous()
        : active_mask.to(torch::kFloat32).contiguous();
#ifdef PULSAR_HAS_PPO_MATH_CUDA_KERNELS
    return normalize_advantage_cuda(advantages.contiguous(), active_arg);
#endif
#ifdef PULSAR_HAS_PPO_MATH_HIP_KERNELS
    return normalize_advantage_hip(advantages.contiguous(), active_arg);
#endif
  }
  const int64_t active_count = active_mask.sum().item<int64_t>();
  if (active_count <= 0) {
    return advantages;
  }
  const torch::Tensor active_adv = advantages.masked_select(active_mask > 0.5F);
  const torch::Tensor mean = active_adv.mean();
  if (active_count <= 1) {
    return clamp_normalized_advantage(advantages - mean);
  }
  const torch::Tensor std = active_adv.std(false).clamp_min(kAdvantageStdFloor);
  return clamp_normalized_advantage((advantages - mean) / std);
}

torch::Tensor sample_future_goal_positions(
    const torch::Tensor& goal_positions,
    const torch::Tensor& dones,
    const torch::Tensor& episode_starts,
    int max_future) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "sample_future_goal_positions");
  if (can_use_ppo_math_accel(goal_positions) &&
      (!dones.defined() || (dones.is_cuda() && dones.scalar_type() == torch::kFloat32)) &&
      (!episode_starts.defined() || (episode_starts.is_cuda() && episode_starts.scalar_type() == torch::kFloat32))) {
    const std::uint32_t seed = get_thread_local_seed();
#ifdef PULSAR_HAS_PPO_MATH_CUDA_KERNELS
    return sample_future_goal_positions_cuda(
        goal_positions.contiguous(),
        dones.defined() ? dones.contiguous() : torch::Tensor{},
        episode_starts.defined() ? episode_starts.contiguous() : torch::Tensor{},
        max_future,
        seed);
#endif
#ifdef PULSAR_HAS_PPO_MATH_HIP_KERNELS
    return sample_future_goal_positions_hip(
        goal_positions.contiguous(),
        dones.defined() ? dones.contiguous() : torch::Tensor{},
        episode_starts.defined() ? episode_starts.contiguous() : torch::Tensor{},
        max_future,
        seed);
#endif
  }
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

  for (int64_t a = 0; a < agents; ++a) {
    int64_t next_done_exclusive = steps;
    int64_t next_start_index = steps;
    for (int64_t t = steps - 1; t >= 0; --t) {
      const int64_t idx = t * agents + a;
      const int64_t done_exclusive = (dones_ptr != nullptr && dones_ptr[idx] > 0.5F)
          ? (t + 1)
          : next_done_exclusive;
      const int64_t end_exclusive = std::min(done_exclusive, next_start_index);
      const int64_t max_offset = std::min<int64_t>(end_exclusive - t - 1, horizon);
      const int64_t chosen = (max_offset > 0)
          ? (t + 1 + get_thread_local_random(max_offset))
          : t;
      for (int64_t d = 0; d < dim; ++d) {
        sampled_cpu[t][a][d] = goal_cpu[chosen][a][d];
      }

      if (dones_ptr != nullptr && dones_ptr[idx] > 0.5F) {
        next_done_exclusive = t + 1;
      }
      if (starts_ptr != nullptr && starts_ptr[idx] > 0.5F) {
        next_start_index = t;
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
    const torch::Tensor& rhs_embeddings,
    float temperature) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "infonce_logits");
  constexpr float kMaxSquaredDistance = 1.0e6F;
  const torch::Tensor lhs = lhs_embeddings.to(torch::kFloat32);
  const torch::Tensor rhs = rhs_embeddings.to(torch::kFloat32);
  const torch::Tensor lhs_norm = lhs.square().sum(-1, true);
  const torch::Tensor rhs_norm = rhs.square().sum(-1, true).transpose(0, 1);
  const torch::Tensor distances = (lhs_norm + rhs_norm - 2.0F * torch::matmul(lhs, rhs.transpose(0, 1)))
      .clamp(1.0e-8F, kMaxSquaredDistance);
  return -distances / std::max(temperature, 1.0e-4F);
}

torch::Tensor compute_symmetric_infonce_loss(
    const torch::Tensor& logits,
    float logsumexp_penalty_coeff) {
  PULSAR_TRACE_SCOPE_CAT("vrpo_math", "infonce_loss");
  constexpr float kFiniteLogitClamp = 1.0e6F;
  const torch::Tensor logits_f32 = logits.to(torch::kFloat32).clamp(-kFiniteLogitClamp, kFiniteLogitClamp);
  const torch::Tensor diag = logits_f32.diagonal();
  const torch::Tensor row_lse = torch::logsumexp(logits_f32, 1);
  const torch::Tensor col_lse = torch::logsumexp(logits_f32, 0);
  const torch::Tensor row_loss = -(diag - row_lse).mean();
  const torch::Tensor col_loss = -(diag - col_lse).mean();
  const torch::Tensor penalty = logsumexp_penalty_coeff * (row_lse.square().mean() + col_lse.square().mean());
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
