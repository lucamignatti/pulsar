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

__global__ void q_boosted_gae_kernel(
    const float* __restrict__ q_values_taken,
    const float* __restrict__ v_from_q,
    const float* __restrict__ rewards,
    const float* __restrict__ dones,
    const float* __restrict__ next_v_from_q,
    const float* __restrict__ bootstrap_mask,
    const float* __restrict__ bootstrap_v_from_q,
    float* __restrict__ advantages,
    int steps,
    int agents,
    float gamma,
    float lambda,
    bool has_next_v_from_q,
    bool has_bootstrap_mask,
    bool has_bootstrap_v_from_q) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= agents) return;
  float last_gae = 0.0F;
  const float boundary = has_next_v_from_q ? next_v_from_q[a] : 0.0F;
  for (int t = steps - 1; t >= 0; --t) {
    const int idx = t * agents + a;
    float next_v = (t + 1 < steps) ? v_from_q[idx + agents] : boundary;
    const float done = dones[idx];
    const float non_terminal = 1.0F - done;
    float delta_mult = non_terminal;
    if (has_bootstrap_mask && bootstrap_mask[idx] > 0.5F) {
      if (has_bootstrap_v_from_q) {
        next_v = bootstrap_v_from_q[idx];
      }
      delta_mult = 1.0F;
    }
    const float delta = rewards[idx] + gamma * next_v * delta_mult - q_values_taken[idx];
    last_gae = delta + gamma * lambda * non_terminal * last_gae;
    advantages[idx] = q_values_taken[idx] - v_from_q[idx] + last_gae;
  }
}

__global__ void centered_expected_sarsa_gae_kernel(
    const float* __restrict__ q_values_taken,
    const float* __restrict__ v_from_q,
    const float* __restrict__ rewards,
    const float* __restrict__ dones,
    const float* __restrict__ next_v_from_q,
    const float* __restrict__ bootstrap_mask,
    const float* __restrict__ bootstrap_v_from_q,
    float* __restrict__ advantages,
    int steps,
    int agents,
    float gamma,
    float lambda,
    bool has_next_v_from_q,
    bool has_bootstrap_mask,
    bool has_bootstrap_v_from_q) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= agents) return;
  float last_gae = 0.0F;
  const float boundary = has_next_v_from_q ? next_v_from_q[a] : 0.0F;
  for (int t = steps - 1; t >= 0; --t) {
    const int idx = t * agents + a;
    float next_v = (t + 1 < steps) ? v_from_q[idx + agents] : boundary;
    const float done = dones[idx];
    const float non_terminal = 1.0F - done;
    float delta_mult = non_terminal;
    if (has_bootstrap_mask && bootstrap_mask[idx] > 0.5F) {
      if (has_bootstrap_v_from_q) {
        next_v = bootstrap_v_from_q[idx];
      }
      delta_mult = 1.0F;
    }
    // Centered residual: subtract V(s) instead of Q(s, a_t).
    const float delta = rewards[idx] + gamma * next_v * delta_mult - v_from_q[idx];
    last_gae = delta + gamma * lambda * non_terminal * last_gae;
    advantages[idx] = q_values_taken[idx] - v_from_q[idx] + last_gae;
  }
}

__global__ void normalize_partial_kernel(
    const float* __restrict__ advantages,
    const float* __restrict__ active,
    double* __restrict__ partial_sum,
    double* __restrict__ partial_sumsq,
    float* __restrict__ partial_count,
    int elements) {
  __shared__ double sums[256];
  __shared__ double sumsqs[256];
  __shared__ float counts[256];
  double sum = 0.0;
  double sumsq = 0.0;
  float count = 0.0F;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x) {
    if (active[i] > 0.5F) {
      const double v = advantages[i];
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
    const double* __restrict__ partial_sum,
    const double* __restrict__ partial_sumsq,
    const float* __restrict__ partial_count,
    float* __restrict__ normalized,
    int elements,
    int partials) {
  __shared__ float stats[3];
  if (threadIdx.x == 0) {
    double sum = 0.0;
    double sumsq = 0.0;
    float count = 0.0F;
    for (int i = 0; i < partials; ++i) {
      sum += partial_sum[i];
      sumsq += partial_sumsq[i];
      count += partial_count[i];
    }
    const double mean = count > 0.0F ? sum / static_cast<double>(count) : 0.0;
    const double var = count > 1.0F ? fmax(sumsq / static_cast<double>(count) - mean * mean, 0.0) : 0.0;
    stats[0] = static_cast<float>(mean);
    stats[1] = count > 1.0F ? static_cast<float>(rsqrt(fmax(var, static_cast<double>(kAdvantageStdFloor * kAdvantageStdFloor)))) : 1.0F;
    stats[2] = count;
  }
  __syncthreads();
  const float mean = stats[0];
  const float inv_std = stats[1];
  const float count = stats[2];
  for (int i = threadIdx.x; i < elements; i += blockDim.x) {
    const float value = count > 1.0F ? (advantages[i] - mean) * inv_std : advantages[i] - mean;
    normalized[i] = fminf(fmaxf(value, -kAdvantageAbsCap), kAdvantageAbsCap);
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

__global__ void sparse_event_soon_targets_kernel(
    const std::uint8_t* __restrict__ events,
    const float* __restrict__ dones,
    const int* __restrict__ horizons,
    int64_t* __restrict__ targets,
    int steps,
    int agents,
    int channels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = steps * agents * channels;
  if (i >= elements) return;
  const int c = i % channels;
  const int n = (i / channels) % agents;
  const int t = i / (agents * channels);
  targets[i] = 0;
  const int horizon = horizons[c] > 1 ? horizons[c] : 1;
  for (int k = 0; k < horizon; ++k) {
    const int look = t + k;
    if (look >= steps) break;
    if (events[(look * agents + n) * channels + c] != 0) {
      targets[i] = 1;
      break;
    }
    if (dones[look * agents + n] > 0.5F) {
      break;
    }
  }
}

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

inline void set_hip_device_for(const at::Tensor& tensor) {
  const int device_index = tensor.get_device();
  if (device_index >= 0) {
    static thread_local int current_device = -2;
    if (current_device != device_index) {
      const hipError_t error = hipSetDevice(device_index);
      TORCH_CHECK(error == hipSuccess, "PPO math HIP set device failed: ", hipGetErrorString(error));
      current_device = device_index;
    }
  }
}

inline void check_hip_launch() {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "PPO math HIP kernel launch failed: ", hipGetErrorString(error));
}

}  // namespace

at::Tensor compute_gae_hip(
    const at::Tensor& values,
    const at::Tensor& rewards,
    const at::Tensor& dones,
    float gamma,
    float gae_lambda,
    const at::Tensor& next_values,
    const at::Tensor& bootstrap_mask,
    const at::Tensor& bootstrap_values) {
  set_hip_device_for(values);
  const int steps = static_cast<int>(values.size(0));
  const int agents = static_cast<int>(values.size(1));
  at::Tensor advantages = at::empty_like(values);
  constexpr int kThreads = 256;
  gae_kernel<<<blocks_for(agents, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
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
  check_hip_launch();
  return advantages;
}

at::Tensor compute_q_boosted_gae_hip(
    const at::Tensor& q_values_taken,
    const at::Tensor& v_from_q,
    const at::Tensor& rewards,
    const at::Tensor& dones,
    float gamma,
    float gae_lambda,
    const at::Tensor& next_v_from_q,
    const at::Tensor& bootstrap_mask,
    const at::Tensor& bootstrap_v_from_q) {
  set_hip_device_for(q_values_taken);
  const int steps = static_cast<int>(q_values_taken.size(0));
  const int agents = static_cast<int>(q_values_taken.size(1));
  at::Tensor advantages = at::empty_like(q_values_taken);
  constexpr int kThreads = 256;
  q_boosted_gae_kernel<<<blocks_for(agents, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      q_values_taken.data_ptr<float>(),
      v_from_q.data_ptr<float>(),
      rewards.data_ptr<float>(),
      dones.data_ptr<float>(),
      next_v_from_q.defined() ? next_v_from_q.data_ptr<float>() : nullptr,
      bootstrap_mask.defined() ? bootstrap_mask.data_ptr<float>() : nullptr,
      bootstrap_v_from_q.defined() ? bootstrap_v_from_q.data_ptr<float>() : nullptr,
      advantages.data_ptr<float>(),
      steps,
      agents,
      gamma,
      gae_lambda,
      next_v_from_q.defined(),
      bootstrap_mask.defined(),
      bootstrap_v_from_q.defined());
  check_hip_launch();
  return advantages;
}

at::Tensor compute_centered_expected_sarsa_gae_hip(
    const at::Tensor& q_values_taken,
    const at::Tensor& v_from_q,
    const at::Tensor& rewards,
    const at::Tensor& dones,
    float gamma,
    float gae_lambda,
    const at::Tensor& next_v_from_q,
    const at::Tensor& bootstrap_mask,
    const at::Tensor& bootstrap_v_from_q) {
  set_hip_device_for(q_values_taken);
  const int steps = static_cast<int>(q_values_taken.size(0));
  const int agents = static_cast<int>(q_values_taken.size(1));
  at::Tensor advantages = at::empty_like(q_values_taken);
  constexpr int kThreads = 256;
  centered_expected_sarsa_gae_kernel<<<blocks_for(agents, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      q_values_taken.data_ptr<float>(),
      v_from_q.data_ptr<float>(),
      rewards.data_ptr<float>(),
      dones.data_ptr<float>(),
      next_v_from_q.defined() ? next_v_from_q.data_ptr<float>() : nullptr,
      bootstrap_mask.defined() ? bootstrap_mask.data_ptr<float>() : nullptr,
      bootstrap_v_from_q.defined() ? bootstrap_v_from_q.data_ptr<float>() : nullptr,
      advantages.data_ptr<float>(),
      steps,
      agents,
      gamma,
      gae_lambda,
      next_v_from_q.defined(),
      bootstrap_mask.defined(),
      bootstrap_v_from_q.defined());
  check_hip_launch();
  return advantages;
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

at::Tensor compute_sparse_event_soon_targets_hip(
    const at::Tensor& sparse_events,
    const at::Tensor& dones,
    const at::Tensor& horizons) {
  set_hip_device_for(sparse_events);
  const int steps = static_cast<int>(sparse_events.size(0));
  const int agents = static_cast<int>(sparse_events.size(1));
  const int channels = static_cast<int>(sparse_events.size(2));
  at::Tensor targets = at::zeros({steps, agents, channels}, sparse_events.options().dtype(at::kLong));
  constexpr int kThreads = 256;
  sparse_event_soon_targets_kernel<<<blocks_for(steps * agents * channels, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      reinterpret_cast<const std::uint8_t*>(sparse_events.data_ptr()),
      dones.data_ptr<float>(),
      horizons.data_ptr<int>(),
      targets.data_ptr<int64_t>(),
      steps,
      agents,
      channels);
  check_hip_launch();
  return targets;
}

}  // namespace pulsar
