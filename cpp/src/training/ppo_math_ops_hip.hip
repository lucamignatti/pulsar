#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

namespace pulsar {
namespace {

constexpr float kAdvantageStdFloor = 1.0F;
constexpr float kAdvantageAbsCap = 10.0F;

__device__ __forceinline__ float safe_logit_exp(float x) {
  return expf(fminf(fmaxf(x, -80.0F), 80.0F));
}

__device__ __forceinline__ std::uint32_t mix_u32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ float uniform01_from_u32(std::uint32_t x) {
  return (static_cast<float>((x >> 8) + 1U)) * (1.0F / 16777217.0F);
}

template <typename scalar_t>
__global__ void sample_masked_actions_kernel(
    const scalar_t* __restrict__ logits,
    const std::uint8_t* __restrict__ masks,
    int64_t* __restrict__ actions,
    float* __restrict__ log_probs,
    int rows,
    int action_dim,
    bool deterministic,
    bool need_log_probs,
    std::uint32_t seed,
    float temperature) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int base = row * action_dim;

  float best = -INFINITY;
  int best_action = 0;
  float max_valid_logit = -INFINITY;
  int valid_count = 0;
  for (int a = 0; a < action_dim; ++a) {
    if (masks[base + a] == 0) continue;
    const float logit = static_cast<float>(logits[base + a]) / temperature;
    ++valid_count;
    if (logit > max_valid_logit) {
      max_valid_logit = logit;
    }
    float score = logit;
    if (!deterministic) {
      const std::uint32_t r = mix_u32(seed ^ static_cast<std::uint32_t>(row * 0x9e3779b9U + a * 0x85ebca6bU));
      const float u = fminf(fmaxf(uniform01_from_u32(r), 1.0e-6F), 1.0F - 1.0e-6F);
      score += -logf(-logf(u));
    }
    if (score > best) {
      best = score;
      best_action = a;
    }
  }
  actions[row] = static_cast<int64_t>(best_action);
  if (need_log_probs) {
    if (valid_count <= 0) {
      log_probs[row] = 0.0F;
      return;
    }
    float sum_exp = 0.0F;
    for (int a = 0; a < action_dim; ++a) {
      if (masks[base + a] != 0) {
        sum_exp += safe_logit_exp(static_cast<float>(logits[base + a]) / temperature - max_valid_logit);
      }
    }
    log_probs[row] = static_cast<float>(logits[base + best_action]) / temperature - (max_valid_logit + logf(fmaxf(sum_exp, 1.0e-20F)));
  }
}

template <typename scalar_t>
__global__ void masked_action_entropy_kernel(
    const scalar_t* __restrict__ logits,
    const std::uint8_t* __restrict__ masks,
    float* __restrict__ entropy,
    int rows,
    int action_dim,
    float temperature) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int base = row * action_dim;
  int valid_count = 0;
  float max_logit = -INFINITY;
  for (int a = 0; a < action_dim; ++a) {
    if (masks[base + a] == 0) continue;
    ++valid_count;
    max_logit = fmaxf(max_logit, static_cast<float>(logits[base + a]) / temperature);
  }
  if (valid_count <= 1) {
    entropy[row] = 0.0F;
    return;
  }
  float sum_exp = 0.0F;
  float weighted = 0.0F;
  for (int a = 0; a < action_dim; ++a) {
    if (masks[base + a] == 0) continue;
    const float shifted = static_cast<float>(logits[base + a]) / temperature - max_logit;
    const float e = safe_logit_exp(shifted);
    sum_exp += e;
    weighted += e * shifted;
  }
  const float log_z = logf(fmaxf(sum_exp, 1.0e-20F));
  const float raw_entropy = log_z - weighted / fmaxf(sum_exp, 1.0e-20F);
  entropy[row] = raw_entropy / fmaxf(logf(static_cast<float>(valid_count)), 1.0e-6F);
}

__global__ void clipped_ppo_loss_forward_kernel(
    const float* __restrict__ current,
    const float* __restrict__ old,
    const float* __restrict__ advantages,
    float* __restrict__ loss,
    int elements,
    float clip_range) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  const float ratio = expf(fminf(fmaxf(current[i] - old[i], -80.0F), 80.0F));
  const float clipped = fminf(fmaxf(ratio, 1.0F - clip_range), 1.0F + clip_range);
  loss[i] = -fminf(ratio * advantages[i], clipped * advantages[i]);
}

__global__ void clipped_ppo_loss_backward_kernel(
    const float* __restrict__ current,
    const float* __restrict__ old,
    const float* __restrict__ advantages,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_current,
    int elements,
    float clip_range) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  const float ratio = expf(fminf(fmaxf(current[i] - old[i], -80.0F), 80.0F));
  const float adv = advantages[i];
  const bool flows = (adv >= 0.0F) ? (ratio <= 1.0F + clip_range) : (ratio >= 1.0F - clip_range);
  grad_current[i] = flows ? (-adv * ratio * grad_output[i]) : 0.0F;
}

__global__ void gae_kernel(
    const float* __restrict__ values,
    const float* __restrict__ rewards,
    const float* __restrict__ dones,
    const float* __restrict__ next_values,
    const float* __restrict__ bootstrap_mask,
    const float* __restrict__ bootstrap_values,
    float* __restrict__ advantages,
    int steps,
    int agents,
    float gamma,
    float lambda,
    bool has_next_values,
    bool has_bootstrap_mask,
    bool has_bootstrap_values) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= agents) return;
  float last_gae = 0.0F;
  const float boundary = has_next_values ? next_values[a] : 0.0F;
  for (int t = steps - 1; t >= 0; --t) {
    const int idx = t * agents + a;
    float next_value = (t + 1 < steps) ? values[idx + agents] : boundary;
    const float done = dones[idx];
    const float non_terminal = 1.0F - done;
    float delta_mult = non_terminal;
    if (has_bootstrap_mask && bootstrap_mask[idx] > 0.5F) {
      if (has_bootstrap_values) {
        next_value = bootstrap_values[idx];
      }
      delta_mult = 1.0F;
    }
    const float delta = rewards[idx] + gamma * next_value * delta_mult - values[idx];
    last_gae = delta + gamma * lambda * non_terminal * last_gae;
    advantages[idx] = last_gae;
  }
}

at::Tensor normalize_advantage_hip(const at::Tensor& advantages, const at::Tensor& active_mask) {
  set_hip_device_for(advantages);
  const int elements = static_cast<int>(advantages.numel());
  at::Tensor normalized = at::empty_like(advantages);
  constexpr int kThreads = 256;
  const int partials = std::min(1024, std::max(1, blocks_for(elements, kThreads)));
  at::Tensor partial_sum = at::empty({partials}, advantages.options().dtype(at::kDouble));
  at::Tensor partial_sumsq = at::empty({partials}, advantages.options().dtype(at::kDouble));
  at::Tensor partial_count = at::empty({partials}, advantages.options());
  normalize_partial_kernel<<<partials, kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      advantages.data_ptr<float>(),
      active_mask.data_ptr<float>(),
      partial_sum.data_ptr<double>(),
      partial_sumsq.data_ptr<double>(),
      partial_count.data_ptr<float>(),
      elements);
  check_hip_launch();
  normalize_finish_kernel<<<1, kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      advantages.data_ptr<float>(),
      partial_sum.data_ptr<double>(),
      partial_sumsq.data_ptr<double>(),
      partial_count.data_ptr<float>(),
      normalized.data_ptr<float>(),
      elements,
      partials);
  check_hip_launch();
  return normalized;
}

std::vector<at::Tensor> sample_masked_actions_hip(
    const at::Tensor& logits,
    const at::Tensor& action_masks,
    bool deterministic,
    bool need_log_probs,
    std::uint32_t seed,
    float temperature) {
  set_hip_device_for(logits);
  const int rows = static_cast<int>(logits.size(0));
  const int action_dim = static_cast<int>(logits.size(1));
  at::Tensor actions = at::empty({rows}, logits.options().dtype(at::kLong));
  at::Tensor log_probs = need_log_probs ? at::empty({rows}, logits.options().dtype(at::kFloat)) : at::Tensor{};
  const at::Tensor masks_u8 = action_masks.to(at::kByte).contiguous();
  const float safe_temperature = fmaxf(temperature, 1.0e-6F);
  constexpr int kThreads = 256;
  if (logits.scalar_type() == at::kHalf) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sample_masked_actions_kernel<at::Half>),
        dim3(blocks_for(rows, kThreads)), dim3(kThreads), 0, at::hip::getCurrentHIPStream(),
        logits.data_ptr<at::Half>(), masks_u8.data_ptr<std::uint8_t>(), actions.data_ptr<int64_t>(),
        need_log_probs ? log_probs.data_ptr<float>() : nullptr, rows, action_dim, deterministic, need_log_probs, seed, safe_temperature);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sample_masked_actions_kernel<float>),
        dim3(blocks_for(rows, kThreads)), dim3(kThreads), 0, at::hip::getCurrentHIPStream(),
        logits.data_ptr<float>(), masks_u8.data_ptr<std::uint8_t>(), actions.data_ptr<int64_t>(),
        need_log_probs ? log_probs.data_ptr<float>() : nullptr, rows, action_dim, deterministic, need_log_probs, seed, safe_temperature);
  }
  check_hip_launch();
  if (need_log_probs) {
    return {actions, log_probs};
  }
  return {actions};
}

at::Tensor masked_action_entropy_hip(const at::Tensor& logits, const at::Tensor& action_masks, float temperature) {
  set_hip_device_for(logits);
  const int rows = static_cast<int>(logits.size(0));
  const int action_dim = static_cast<int>(logits.size(1));
  at::Tensor entropy = at::empty({rows}, logits.options().dtype(at::kFloat));
  const at::Tensor masks_u8 = action_masks.to(at::kByte).contiguous();
  const float safe_temperature = fmaxf(temperature, 1.0e-6F);
  constexpr int kThreads = 256;
  if (logits.scalar_type() == at::kHalf) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(masked_action_entropy_kernel<at::Half>),
        dim3(blocks_for(rows, kThreads)), dim3(kThreads), 0, at::hip::getCurrentHIPStream(),
        logits.data_ptr<at::Half>(), masks_u8.data_ptr<std::uint8_t>(), entropy.data_ptr<float>(), rows, action_dim, safe_temperature);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(masked_action_entropy_kernel<float>),
        dim3(blocks_for(rows, kThreads)), dim3(kThreads), 0, at::hip::getCurrentHIPStream(),
        logits.data_ptr<float>(), masks_u8.data_ptr<std::uint8_t>(), entropy.data_ptr<float>(), rows, action_dim, safe_temperature);
  }
  check_hip_launch();
  return entropy;
}

at::Tensor clipped_ppo_policy_loss_forward_hip(
    const at::Tensor& current_log_probs,
    const at::Tensor& old_log_probs,
    const at::Tensor& advantages,
    float clip_range) {
  set_hip_device_for(current_log_probs);
  at::Tensor loss = at::empty_like(current_log_probs);
  const int elements = static_cast<int>(current_log_probs.numel());
  constexpr int kThreads = 256;
  clipped_ppo_loss_forward_kernel<<<blocks_for(elements, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      current_log_probs.data_ptr<float>(),
      old_log_probs.data_ptr<float>(),
      advantages.data_ptr<float>(),
      loss.data_ptr<float>(),
      elements,
      clip_range);
  check_hip_launch();
  return loss;
}

at::Tensor clipped_ppo_policy_loss_backward_hip(
    const at::Tensor& current_log_probs,
    const at::Tensor& old_log_probs,
    const at::Tensor& advantages,
    const at::Tensor& grad_output,
    float clip_range) {
  set_hip_device_for(current_log_probs);
  at::Tensor grad_current = at::empty_like(current_log_probs);
  const int elements = static_cast<int>(current_log_probs.numel());
  constexpr int kThreads = 256;
  clipped_ppo_loss_backward_kernel<<<blocks_for(elements, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      current_log_probs.data_ptr<float>(),
      old_log_probs.data_ptr<float>(),
      advantages.data_ptr<float>(),
      grad_output.data_ptr<float>(),
      grad_current.data_ptr<float>(),
      elements,
      clip_range);
  check_hip_launch();
  return grad_current;
}

at::Tensor sample_future_goal_positions_hip(
    const at::Tensor& goal_positions,
    const at::Tensor& dones,
    const at::Tensor& episode_starts,
    int max_future,
    std::uint32_t seed) {
  set_hip_device_for(goal_positions);
  const int steps = static_cast<int>(goal_positions.size(0));
  const int agents = static_cast<int>(goal_positions.size(1));
  const int dim = static_cast<int>(goal_positions.size(2));
  at::Tensor sampled = at::empty_like(goal_positions);
  constexpr int kThreads = 256;
  sample_future_goal_kernel<<<blocks_for(steps * agents, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      goal_positions.data_ptr<float>(),
      dones.defined() ? dones.data_ptr<float>() : nullptr,
      episode_starts.defined() ? episode_starts.data_ptr<float>() : nullptr,
      sampled.data_ptr<float>(),
      steps,
      agents,
      dim,
      std::max(1, max_future),
      seed);
  check_hip_launch();
  return sampled;
}

}  // namespace pulsar
