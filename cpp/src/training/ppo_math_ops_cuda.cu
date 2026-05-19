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

__device__ __forceinline__ std::uint32_t mix_u32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
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

__global__ void normalize_partial_kernel(
    const float* __restrict__ advantages,
    const float* __restrict__ active,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sumsq,
    float* __restrict__ partial_count,
    int elements) {
  __shared__ float sums[256];
  __shared__ float sumsqs[256];
  __shared__ float counts[256];
  float sum = 0.0F;
  float sumsq = 0.0F;
  float count = 0.0F;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x) {
    if (active[i] > 0.5F) {
      const float v = advantages[i];
      sum += v;
      sumsq += v * v;
      count += 1.0F;
    }
  }
  sums[threadIdx.x] = sum;
  sumsqs[threadIdx.x] = sumsq;
  counts[threadIdx.x] = count;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
      sumsqs[threadIdx.x] += sumsqs[threadIdx.x + stride];
      counts[threadIdx.x] += counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_sum[blockIdx.x] = sums[0];
    partial_sumsq[blockIdx.x] = sumsqs[0];
    partial_count[blockIdx.x] = counts[0];
  }
}

__global__ void normalize_finish_kernel(
    const float* __restrict__ advantages,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_sumsq,
    const float* __restrict__ partial_count,
    float* __restrict__ normalized,
    int elements,
    int partials) {
  __shared__ float stats[3];
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    float sumsq = 0.0F;
    float count = 0.0F;
    for (int i = 0; i < partials; ++i) {
      sum += partial_sum[i];
      sumsq += partial_sumsq[i];
      count += partial_count[i];
    }
    const float mean = count > 0.0F ? sum / count : 0.0F;
    const float var = count > 1.0F ? fmaxf(sumsq / count - mean * mean, 0.0F) : 0.0F;
    stats[0] = mean;
    stats[1] = count > 1.0F ? rsqrtf(fmaxf(var, 1.0e-16F)) : 1.0F;
    stats[2] = count;
  }
  __syncthreads();
  const float mean = stats[0];
  const float inv_std = stats[1];
  const float count = stats[2];
  for (int i = threadIdx.x; i < elements; i += blockDim.x) {
    normalized[i] = count > 1.0F ? (advantages[i] - mean) * inv_std : advantages[i] - mean;
  }
}

__global__ void sample_future_goal_kernel(
    const float* __restrict__ goal_positions,
    const float* __restrict__ dones,
    const float* __restrict__ starts,
    float* __restrict__ sampled,
    int steps,
    int agents,
    int dim,
    int horizon,
    std::uint32_t seed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = steps * agents;
  if (index >= total) return;
  const int t = index / agents;
  const int a = index - t * agents;
  int end = steps;
  for (int j = t; j < steps; ++j) {
    const int ja = j * agents + a;
    if (dones != nullptr && dones[ja] > 0.5F) {
      end = j + 1;
      break;
    }
    if (starts != nullptr && starts[ja] > 0.5F && j > t) {
      end = j;
      break;
    }
  }
  int max_offset = end - t - 1;
  if (max_offset < 0) max_offset = 0;
  if (max_offset > horizon) max_offset = horizon;
  int chosen = t;
  if (max_offset > 0) {
    const std::uint32_t r = mix_u32(seed ^ static_cast<std::uint32_t>(index * 747796405U));
    chosen = t + 1 + static_cast<int>(r % static_cast<std::uint32_t>(max_offset));
  }
  const int src_base = (chosen * agents + a) * dim;
  const int dst_base = (t * agents + a) * dim;
  for (int d = 0; d < dim; ++d) {
    sampled[dst_base + d] = goal_positions[src_base + d];
  }
}

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

}  // namespace

at::Tensor compute_gae_cuda(
    const at::Tensor& values,
    const at::Tensor& rewards,
    const at::Tensor& dones,
    float gamma,
    float gae_lambda,
    const at::Tensor& next_values,
    const at::Tensor& bootstrap_mask,
    const at::Tensor& bootstrap_values) {
  const c10::cuda::CUDAGuard guard(values.device());
  const int steps = static_cast<int>(values.size(0));
  const int agents = static_cast<int>(values.size(1));
  at::Tensor advantages = at::empty_like(values);
  constexpr int kThreads = 256;
  gae_kernel<<<blocks_for(agents, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<float>(),
      rewards.data_ptr<float>(),
      dones.data_ptr<float>(),
      next_values.defined() ? next_values.data_ptr<float>() : nullptr,
      bootstrap_mask.defined() ? bootstrap_mask.data_ptr<float>() : nullptr,
      bootstrap_values.defined() ? bootstrap_values.data_ptr<float>() : nullptr,
      advantages.data_ptr<float>(),
      steps,
      agents,
      gamma,
      gae_lambda,
      next_values.defined(),
      bootstrap_mask.defined(),
      bootstrap_values.defined());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return advantages;
}

at::Tensor normalize_advantage_cuda(const at::Tensor& advantages, const at::Tensor& active_mask) {
  const c10::cuda::CUDAGuard guard(advantages.device());
  const int elements = static_cast<int>(advantages.numel());
  at::Tensor normalized = at::empty_like(advantages);
  constexpr int kThreads = 256;
  const int partials = std::min(1024, std::max(1, blocks_for(elements, kThreads)));
  at::Tensor partial_sum = at::empty({partials}, advantages.options());
  at::Tensor partial_sumsq = at::empty({partials}, advantages.options());
  at::Tensor partial_count = at::empty({partials}, advantages.options());
  normalize_partial_kernel<<<partials, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      advantages.data_ptr<float>(),
      active_mask.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      partial_sumsq.data_ptr<float>(),
      partial_count.data_ptr<float>(),
      elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  normalize_finish_kernel<<<1, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      advantages.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      partial_sumsq.data_ptr<float>(),
      partial_count.data_ptr<float>(),
      normalized.data_ptr<float>(),
      elements,
      partials);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return normalized;
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

}  // namespace pulsar
