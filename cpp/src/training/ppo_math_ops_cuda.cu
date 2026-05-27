#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace pulsar {
namespace {

constexpr float kAdvantageStdFloor = 1.0F;
constexpr float kAdvantageAbsCap = 10.0F;

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

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

__global__ void normalize_partial_kernel(
    const float* __restrict__ advantages,
    const float* __restrict__ active_mask,
    double* __restrict__ partial_sum,
    double* __restrict__ partial_sumsq,
    float* __restrict__ partial_count,
    int elements) {
  __shared__ double block_sum[256];
  __shared__ double block_sumsq[256];
  __shared__ float block_count[256];

  const int tid = threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  double local_sum = 0.0;
  double local_sumsq = 0.0;
  float local_count = 0.0F;
  for (int i = blockIdx.x * blockDim.x + tid; i < elements; i += stride) {
    if (active_mask[i] > 0.5F) {
      const double value = static_cast<double>(advantages[i]);
      local_sum += value;
      local_sumsq += value * value;
      local_count += 1.0F;
    }
  }

  block_sum[tid] = local_sum;
  block_sumsq[tid] = local_sumsq;
  block_count[tid] = local_count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      block_sum[tid] += block_sum[tid + stride];
      block_sumsq[tid] += block_sumsq[tid + stride];
      block_count[tid] += block_count[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sum[blockIdx.x] = block_sum[0];
    partial_sumsq[blockIdx.x] = block_sumsq[0];
    partial_count[blockIdx.x] = block_count[0];
  }
}

__global__ void normalize_finish_kernel(
    const float* __restrict__ advantages,
    const double* __restrict__ partial_sum,
    const double* __restrict__ partial_sumsq,
    const float* __restrict__ partial_count,
    float* __restrict__ normalized,
    int elements,
    int partials) {
  __shared__ double mean;
  __shared__ double inv_std;

  if (threadIdx.x == 0) {
    double total_sum = 0.0;
    double total_sumsq = 0.0;
    double total_count = 0.0;
    for (int p = 0; p < partials; ++p) {
      total_sum += partial_sum[p];
      total_sumsq += partial_sumsq[p];
      total_count += static_cast<double>(partial_count[p]);
    }
    if (total_count <= 0.0) {
      mean = 0.0;
      inv_std = 0.0;
    } else {
      mean = total_sum / total_count;
      const double variance = fmax(total_sumsq / total_count - mean * mean, 0.0);
      const double std_value = sqrt(fmax(variance, static_cast<double>(kAdvantageStdFloor * kAdvantageStdFloor)));
      inv_std = 1.0 / std_value;
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < elements; i += blockDim.x) {
    if (inv_std == 0.0) {
      normalized[i] = advantages[i];
      continue;
    }
    const double value = static_cast<double>(advantages[i]);
    const float normed = static_cast<float>((value - mean) * inv_std);
    normalized[i] = fmaxf(-kAdvantageAbsCap, fminf(kAdvantageAbsCap, normed));
  }
}

__global__ void sample_future_goal_kernel(
    const float* __restrict__ goal_positions,
    const float* __restrict__ dones,
    const float* __restrict__ episode_starts,
    float* __restrict__ sampled,
    int steps,
    int agents,
    int dim,
    int max_future,
    std::uint32_t seed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= steps * agents) return;

  const int t = idx / agents;
  const int a = idx % agents;

  int horizon = 0;
  for (int future_t = t + 1; future_t < steps && future_t <= t + max_future; ++future_t) {
    const int future_idx = future_t * agents + a;
    if (episode_starts != nullptr && episode_starts[future_idx] > 0.5F) break;
    ++horizon;
    if (dones != nullptr && dones[future_idx] > 0.5F) break;
  }

  const int offset = (horizon > 0)
      ? (1 + static_cast<int>(mix_u32(seed ^ static_cast<std::uint32_t>(idx) * 0x9e3779b9U) % static_cast<std::uint32_t>(horizon)))
      : 0;
  const int src_idx = (t + offset) * agents + a;
  const int dst_base = idx * dim;
  const int src_base = src_idx * dim;
  for (int d = 0; d < dim; ++d) {
    sampled[dst_base + d] = goal_positions[src_base + d];
  }
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

at::Tensor normalize_advantage_cuda(const at::Tensor& advantages, const at::Tensor& active_mask) {
  const c10::cuda::CUDAGuard guard(advantages.device());
  const int elements = static_cast<int>(advantages.numel());
  at::Tensor normalized = at::empty_like(advantages);
  constexpr int kThreads = 256;
  const int partials = std::min(1024, std::max(1, blocks_for(elements, kThreads)));
  at::Tensor partial_sum = at::empty({partials}, advantages.options().dtype(at::kDouble));
  at::Tensor partial_sumsq = at::empty({partials}, advantages.options().dtype(at::kDouble));
  at::Tensor partial_count = at::empty({partials}, advantages.options());
  normalize_partial_kernel<<<partials, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      advantages.data_ptr<float>(),
      active_mask.data_ptr<float>(),
      partial_sum.data_ptr<double>(),
      partial_sumsq.data_ptr<double>(),
      partial_count.data_ptr<float>(),
      elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  normalize_finish_kernel<<<1, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      advantages.data_ptr<float>(),
      partial_sum.data_ptr<double>(),
      partial_sumsq.data_ptr<double>(),
      partial_count.data_ptr<float>(),
      normalized.data_ptr<float>(),
      elements,
      partials);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return normalized;
}

std::vector<at::Tensor> sample_masked_actions_cuda(
    const at::Tensor& logits,
    const at::Tensor& action_masks,
    bool deterministic,
    bool need_log_probs,
    std::uint32_t seed,
    float temperature) {
  const c10::cuda::CUDAGuard guard(logits.device());
  const int rows = static_cast<int>(logits.size(0));
  const int action_dim = static_cast<int>(logits.size(1));
  at::Tensor actions = at::empty({rows}, logits.options().dtype(at::kLong));
  at::Tensor log_probs = need_log_probs ? at::empty({rows}, logits.options().dtype(at::kFloat)) : at::Tensor{};
  const at::Tensor masks_u8 = action_masks.to(at::kByte).contiguous();
  const float safe_temperature = fmaxf(temperature, 1.0e-6F);
  constexpr int kThreads = 256;
  if (logits.scalar_type() == at::kHalf) {
    sample_masked_actions_kernel<at::Half><<<blocks_for(rows, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<at::Half>(), masks_u8.data_ptr<std::uint8_t>(), actions.data_ptr<int64_t>(),
        need_log_probs ? log_probs.data_ptr<float>() : nullptr, rows, action_dim, deterministic, need_log_probs, seed, safe_temperature);
  } else {
    sample_masked_actions_kernel<float><<<blocks_for(rows, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), masks_u8.data_ptr<std::uint8_t>(), actions.data_ptr<int64_t>(),
        need_log_probs ? log_probs.data_ptr<float>() : nullptr, rows, action_dim, deterministic, need_log_probs, seed, safe_temperature);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (need_log_probs) {
    return {actions, log_probs};
  }
  return {actions};
}

at::Tensor masked_action_entropy_cuda(const at::Tensor& logits, const at::Tensor& action_masks, float temperature) {
  const c10::cuda::CUDAGuard guard(logits.device());
  const int rows = static_cast<int>(logits.size(0));
  const int action_dim = static_cast<int>(logits.size(1));
  at::Tensor entropy = at::empty({rows}, logits.options().dtype(at::kFloat));
  const at::Tensor masks_u8 = action_masks.to(at::kByte).contiguous();
  const float safe_temperature = fmaxf(temperature, 1.0e-6F);
  constexpr int kThreads = 256;
  if (logits.scalar_type() == at::kHalf) {
    masked_action_entropy_kernel<at::Half><<<blocks_for(rows, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<at::Half>(), masks_u8.data_ptr<std::uint8_t>(), entropy.data_ptr<float>(), rows, action_dim, safe_temperature);
  } else {
    masked_action_entropy_kernel<float><<<blocks_for(rows, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), masks_u8.data_ptr<std::uint8_t>(), entropy.data_ptr<float>(), rows, action_dim, safe_temperature);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return entropy;
}

at::Tensor clipped_ppo_policy_loss_forward_cuda(
    const at::Tensor& current_log_probs,
    const at::Tensor& old_log_probs,
    const at::Tensor& advantages,
    float clip_range) {
  const c10::cuda::CUDAGuard guard(current_log_probs.device());
  at::Tensor loss = at::empty_like(current_log_probs);
  const int elements = static_cast<int>(current_log_probs.numel());
  constexpr int kThreads = 256;
  clipped_ppo_loss_forward_kernel<<<blocks_for(elements, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      current_log_probs.data_ptr<float>(),
      old_log_probs.data_ptr<float>(),
      advantages.data_ptr<float>(),
      loss.data_ptr<float>(),
      elements,
      clip_range);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return loss;
}

at::Tensor clipped_ppo_policy_loss_backward_cuda(
    const at::Tensor& current_log_probs,
    const at::Tensor& old_log_probs,
    const at::Tensor& advantages,
    const at::Tensor& grad_output,
    float clip_range) {
  const c10::cuda::CUDAGuard guard(current_log_probs.device());
  at::Tensor grad_current = at::empty_like(current_log_probs);
  const int elements = static_cast<int>(current_log_probs.numel());
  constexpr int kThreads = 256;
  clipped_ppo_loss_backward_kernel<<<blocks_for(elements, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      current_log_probs.data_ptr<float>(),
      old_log_probs.data_ptr<float>(),
      advantages.data_ptr<float>(),
      grad_output.data_ptr<float>(),
      grad_current.data_ptr<float>(),
      elements,
      clip_range);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_current;
}

at::Tensor sample_future_goal_positions_cuda(
    const at::Tensor& goal_positions,
    const at::Tensor& dones,
    const at::Tensor& episode_starts,
    int max_future,
    std::uint32_t seed) {
  const c10::cuda::CUDAGuard guard(goal_positions.device());
  const int steps = static_cast<int>(goal_positions.size(0));
  const int agents = static_cast<int>(goal_positions.size(1));
  const int dim = static_cast<int>(goal_positions.size(2));
  at::Tensor sampled = at::empty_like(goal_positions);
  constexpr int kThreads = 256;
  sample_future_goal_kernel<<<blocks_for(steps * agents, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      goal_positions.data_ptr<float>(),
      dones.defined() ? dones.data_ptr<float>() : nullptr,
      episode_starts.defined() ? episode_starts.data_ptr<float>() : nullptr,
      sampled.data_ptr<float>(),
      steps,
      agents,
      dim,
      std::max(1, max_future),
      seed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sampled;
}

}  // namespace
}  // namespace pulsar
