#include <tuple>
#include <vector>
#include <cstring>
#include <cstdlib>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

namespace pulsar {
namespace {

constexpr float kRetentionMin = 0.01F;
constexpr float kRetentionMax = 0.9999F;

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.0F / (1.0F + expf(-x));
}

__device__ __forceinline__ float siluf_fast(float x) {
  return x * sigmoidf_fast(x);
}

__device__ __forceinline__ float silu_gradf_fast(float x) {
  const float s = sigmoidf_fast(x);
  return s * (1.0F + x * (1.0F - s));
}

template <bool HasReset>
__global__ void mamba2_scan_forward_kernel(
    const float* __restrict__ projected,
    const float* __restrict__ decay_bias,
    const float* __restrict__ skip,
    const float* __restrict__ reset,
    float* __restrict__ mixed,
    float* __restrict__ scan,
    int batch,
    int sequence,
    int embed_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * embed_dim;
  if (index >= total) return;

  const int b = index / embed_dim;
  const int d = index - b * embed_dim;
  const int projected_stride = 5 * embed_dim;
  int base = b * sequence * projected_stride + d;
  int out = b * sequence * embed_dim + d;
  const int reset_base = b * sequence;
  float state = 0.0F;
  for (int t = 0; t < sequence; ++t) {
    float keep = 1.0F;
    if constexpr (HasReset) {
      keep = reset[reset_base + t] > 0.5F ? 0.0F : 1.0F;
    }
    state *= keep;
    const float p0 = projected[base];
    const float p1 = projected[base + embed_dim];
    const float p2 = projected[base + 2 * embed_dim];
    const float p3 = projected[base + 3 * embed_dim];
    const float p4 = projected[base + 4 * embed_dim] + decay_bias[d];
    const float x = siluf_fast(p0);
    const float gate_b = sigmoidf_fast(p1);
    const float gate_c = sigmoidf_fast(p2);
    const float gate_z = siluf_fast(p3);
    const float retention = fminf(fmaxf(sigmoidf_fast(p4), kRetentionMin), kRetentionMax);
    state = retention * state + gate_b * x;
    scan[out] = state;
    mixed[out] = (gate_c * state + skip[d] * x) * gate_z;
    base += projected_stride;
    out += embed_dim;
  }
}

template <bool HasReset, bool DirectReduce>
__global__ void mamba2_scan_backward_kernel(
    const float* __restrict__ grad_mixed,
    const float* __restrict__ projected,
    const float* __restrict__ decay_bias,
    const float* __restrict__ skip,
    const float* __restrict__ reset,
    const float* __restrict__ scan,
    float* __restrict__ grad_projected,
    float* __restrict__ grad_decay_out,
    float* __restrict__ grad_skip_out,
    int batch,
    int sequence,
    int embed_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * embed_dim;
  if (index >= total) return;

  const int b = index / embed_dim;
  const int d = index - b * embed_dim;
  const int projected_stride = 5 * embed_dim;
  const int reset_base = b * sequence;
  float grad_next_scan = 0.0F;
  float local_grad_skip = 0.0F;
  float local_grad_decay = 0.0F;

  for (int t = sequence - 1; t >= 0; --t) {
    const int proj_base = ((b * sequence + t) * projected_stride) + d;
    const int out = (b * sequence + t) * embed_dim + d;
    float keep = 1.0F;
    if constexpr (HasReset) {
      keep = reset[reset_base + t] > 0.5F ? 0.0F : 1.0F;
    }
    const float prev_scan = (t > 0) ? scan[((b * sequence + (t - 1)) * embed_dim) + d] : 0.0F;
    const float p0 = projected[proj_base];
    const float p1 = projected[proj_base + embed_dim];
    const float p2 = projected[proj_base + 2 * embed_dim];
    const float p3 = projected[proj_base + 3 * embed_dim];
    const float p4_raw = projected[proj_base + 4 * embed_dim] + decay_bias[d];
    const float x = siluf_fast(p0);
    const float gate_b = sigmoidf_fast(p1);
    const float gate_c = sigmoidf_fast(p2);
    const float gate_z = siluf_fast(p3);
    const float retention_unclamped = sigmoidf_fast(p4_raw);
    const float retention = fminf(fmaxf(retention_unclamped, kRetentionMin), kRetentionMax);
    const float state = scan[out];
    const float mixed_pre = gate_c * state + skip[d] * x;
    const float gy = grad_mixed[out];
    const float grad_z = gy * mixed_pre;
    const float grad_mixed_pre = gy * gate_z;
    const float grad_c = grad_mixed_pre * state;
    const float grad_state = grad_mixed_pre * gate_c + grad_next_scan;
    local_grad_skip += grad_mixed_pre * x;
    float grad_x = grad_mixed_pre * skip[d];
    const float grad_b = grad_state * x;
    grad_x += grad_state * gate_b;
    const float grad_retention = grad_state * keep * prev_scan;
    grad_next_scan = grad_state * retention * keep;
    grad_projected[proj_base] = grad_x * silu_gradf_fast(p0);
    grad_projected[proj_base + embed_dim] = grad_b * gate_b * (1.0F - gate_b);
    grad_projected[proj_base + 2 * embed_dim] = grad_c * gate_c * (1.0F - gate_c);
    grad_projected[proj_base + 3 * embed_dim] = grad_z * silu_gradf_fast(p3);
    const float active = (retention_unclamped > kRetentionMin && retention_unclamped < kRetentionMax) ? 1.0F : 0.0F;
    const float grad_retention_pre = grad_retention * active * retention_unclamped * (1.0F - retention_unclamped);
    grad_projected[proj_base + 4 * embed_dim] = grad_retention_pre;
    local_grad_decay += grad_retention_pre;
  }
  if constexpr (DirectReduce) {
    atomicAdd(grad_skip_out + d, local_grad_skip);
    atomicAdd(grad_decay_out + d, local_grad_decay);
  } else {
    const int partial_index = b * embed_dim + d;
    grad_skip_out[partial_index] = local_grad_skip;
    grad_decay_out[partial_index] = local_grad_decay;
  }
}

__global__ void reduce_batch_dim_pair_kernel(
    const float* __restrict__ left_partial,
    const float* __restrict__ right_partial,
    float* __restrict__ left_reduced,
    float* __restrict__ right_reduced,
    int batch,
    int embed_dim) {
  const int d = blockIdx.x;
  if (d >= embed_dim) return;
  __shared__ float left_sums[256];
  __shared__ float right_sums[256];
  float left_sum = 0.0F;
  float right_sum = 0.0F;
  for (int b = threadIdx.x; b < batch; b += blockDim.x) {
    const int index = b * embed_dim + d;
    left_sum += left_partial[index];
    right_sum += right_partial[index];
  }
  left_sums[threadIdx.x] = left_sum;
  right_sums[threadIdx.x] = right_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      left_sums[threadIdx.x] += left_sums[threadIdx.x + stride];
      right_sums[threadIdx.x] += right_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    left_reduced[d] = left_sums[0];
    right_reduced[d] = right_sums[0];
  }
}

__global__ void mamba2_step_forward_kernel(
    const float* __restrict__ projected,
    const float* __restrict__ previous_scan,
    const float* __restrict__ decay_bias,
    const float* __restrict__ skip,
    float* __restrict__ mixed,
    float* __restrict__ next_scan,
    int batch,
    int embed_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * embed_dim;
  if (index >= total) return;
  const int b = index / embed_dim;
  const int d = index - b * embed_dim;
  const int base = b * 5 * embed_dim + d;
  const float p0 = projected[base];
  const float p1 = projected[base + embed_dim];
  const float p2 = projected[base + 2 * embed_dim];
  const float p3 = projected[base + 3 * embed_dim];
  const float p4 = projected[base + 4 * embed_dim] + decay_bias[d];
  const float x = siluf_fast(p0);
  const float gate_b = sigmoidf_fast(p1);
  const float gate_c = sigmoidf_fast(p2);
  const float gate_z = siluf_fast(p3);
  const float retention = fminf(fmaxf(sigmoidf_fast(p4), kRetentionMin), kRetentionMax);
  const float scan = retention * previous_scan[b * embed_dim + d] + gate_b * x;
  next_scan[b * embed_dim + d] = scan;
  mixed[b * embed_dim + d] = (gate_c * scan + skip[d] * x) * gate_z;
}

__global__ void causal_conv1d_silu_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ reset,
    float* __restrict__ output,
    float* __restrict__ pre_activation,
    int batch,
    int sequence,
    int embed_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * sequence * embed_dim;
  if (index >= total) return;
  const int d = index % embed_dim;
  const int t = (index / embed_dim) % sequence;
  const int b = index / (sequence * embed_dim);
  const int reset_base = b * sequence;
  const int current = (b * sequence + t) * embed_dim + d;
  const float keep1 = (t > 0 && reset[reset_base + t] <= 0.5F) ? 1.0F : 0.0F;
  const float keep2 =
      (t > 1 && reset[reset_base + t] <= 0.5F && reset[reset_base + t - 1] <= 0.5F) ? 1.0F : 0.0F;
  const float prev1 = keep1 > 0.0F ? input[current - embed_dim] : 0.0F;
  const float prev2 = keep2 > 0.0F ? input[current - 2 * embed_dim] : 0.0F;
  const float pre = prev2 * weight[d * 3] + prev1 * weight[d * 3 + 1] + input[current] * weight[d * 3 + 2] + bias[d];
  pre_activation[index] = pre;
  output[index] = siluf_fast(pre);
}

__global__ void causal_conv1d_silu_backward_input_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ pre_activation,
    const float* __restrict__ weight,
    const float* __restrict__ reset,
    float* __restrict__ grad_input,
    int batch,
    int sequence,
    int embed_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * sequence * embed_dim;
  if (index >= total) return;
  const int d = index % embed_dim;
  const int t = (index / embed_dim) % sequence;
  const int b = index / (sequence * embed_dim);
  const int reset_base = b * sequence;
  const int current = (b * sequence + t) * embed_dim + d;
  float grad = grad_output[current] * silu_gradf_fast(pre_activation[current]) * weight[d * 3 + 2];
  if (t + 1 < sequence && reset[reset_base + t + 1] <= 0.5F) {
    const int next = current + embed_dim;
    grad += grad_output[next] * silu_gradf_fast(pre_activation[next]) * weight[d * 3 + 1];
  }
  if (t + 2 < sequence && reset[reset_base + t + 2] <= 0.5F && reset[reset_base + t + 1] <= 0.5F) {
    const int next2 = current + 2 * embed_dim;
    grad += grad_output[next2] * silu_gradf_fast(pre_activation[next2]) * weight[d * 3];
  }
  grad_input[current] = grad;
}

__global__ void causal_conv1d_silu_backward_params_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ pre_activation,
    const float* __restrict__ reset,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int batch,
    int sequence,
    int embed_dim) {
  const int d = blockIdx.x;
  if (d >= embed_dim) return;
  __shared__ float sums0[256];
  __shared__ float sums1[256];
  __shared__ float sums2[256];
  __shared__ float sums_bias[256];
  float sum0 = 0.0F;
  float sum1 = 0.0F;
  float sum2 = 0.0F;
  float sum_bias = 0.0F;
  const int total = batch * sequence;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    const int b = i / sequence;
    const int t = i - b * sequence;
    const int reset_base = b * sequence;
    const int current = (b * sequence + t) * embed_dim + d;
    const float g = grad_output[current] * silu_gradf_fast(pre_activation[current]);
    if (t > 1 && reset[reset_base + t] <= 0.5F && reset[reset_base + t - 1] <= 0.5F) {
      sum0 += g * input[current - 2 * embed_dim];
    }
    if (t > 0 && reset[reset_base + t] <= 0.5F) {
      sum1 += g * input[current - embed_dim];
    }
    sum2 += g * input[current];
    sum_bias += g;
  }
  sums0[threadIdx.x] = sum0;
  sums1[threadIdx.x] = sum1;
  sums2[threadIdx.x] = sum2;
  sums_bias[threadIdx.x] = sum_bias;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums0[threadIdx.x] += sums0[threadIdx.x + stride];
      sums1[threadIdx.x] += sums1[threadIdx.x + stride];
      sums2[threadIdx.x] += sums2[threadIdx.x + stride];
      sums_bias[threadIdx.x] += sums_bias[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    grad_weight[d * 3] = sums0[0];
    grad_weight[d * 3 + 1] = sums1[0];
    grad_weight[d * 3 + 2] = sums2[0];
    grad_bias[d] = sums_bias[0];
  }
}

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

inline void set_hip_device_for(const at::Tensor& tensor) {
  const int device_index = tensor.get_device();
  if (device_index >= 0) {
    const hipError_t error = hipSetDevice(device_index);
    TORCH_CHECK(error == hipSuccess, "Mamba2 HIP set device failed: ", hipGetErrorString(error));
  }
}

inline void check_hip_launch() {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "Mamba2 HIP kernel launch failed: ", hipGetErrorString(error));
}

inline bool use_direct_grad_reduce() {
  static const bool direct_reduce = [] {
    const char* value = std::getenv("PULSAR_MAMBA2_GRAD_REDUCE");
    return value == nullptr || std::strcmp(value, "pair") != 0;
  }();
  return direct_reduce;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> mamba2_scan_forward_hip(
    const at::Tensor& projected,
    const at::Tensor& decay_bias,
    const at::Tensor& skip,
    const at::Tensor& reset_mask) {
  set_hip_device_for(projected);
  const int batch = static_cast<int>(projected.size(0));
  const int sequence = static_cast<int>(projected.size(1));
  const int embed_dim = static_cast<int>(projected.size(2) / 5);
  const bool has_reset = reset_mask.defined();
  at::Tensor mixed = at::empty({batch, sequence, embed_dim}, projected.options());
  at::Tensor scan = at::empty({batch, sequence, embed_dim}, projected.options());
  constexpr int kThreads = 256;
  const int elements = batch * embed_dim;
  if (has_reset) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mamba2_scan_forward_kernel<true>),
        dim3(blocks_for(elements, kThreads)),
        dim3(kThreads),
        0,
        at::hip::getCurrentHIPStream(),
        projected.data_ptr<float>(),
        decay_bias.data_ptr<float>(),
        skip.data_ptr<float>(),
        reset_mask.data_ptr<float>(),
        mixed.data_ptr<float>(),
        scan.data_ptr<float>(),
        batch,
        sequence,
        embed_dim);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mamba2_scan_forward_kernel<false>),
        dim3(blocks_for(elements, kThreads)),
        dim3(kThreads),
        0,
        at::hip::getCurrentHIPStream(),
        projected.data_ptr<float>(),
        decay_bias.data_ptr<float>(),
        skip.data_ptr<float>(),
        nullptr,
        mixed.data_ptr<float>(),
        scan.data_ptr<float>(),
        batch,
        sequence,
        embed_dim);
  }
  check_hip_launch();
  return {mixed, scan};
}

std::vector<at::Tensor> mamba2_scan_backward_hip(
    const at::Tensor& grad_mixed,
    const at::Tensor& projected,
    const at::Tensor& decay_bias,
    const at::Tensor& skip,
    const at::Tensor& reset_mask,
    const at::Tensor& scan) {
  set_hip_device_for(projected);
  const int batch = static_cast<int>(projected.size(0));
  const int sequence = static_cast<int>(projected.size(1));
  const int embed_dim = static_cast<int>(projected.size(2) / 5);
  const bool has_reset = reset_mask.defined();
  at::Tensor grad_projected = at::empty_like(projected);
  const bool direct_reduce = use_direct_grad_reduce();
  at::Tensor grad_decay_bias = at::empty_like(decay_bias);
  at::Tensor grad_skip = at::empty_like(skip);
  at::Tensor grad_decay_partial = direct_reduce ? at::Tensor{} : at::empty({batch, embed_dim}, projected.options());
  at::Tensor grad_skip_partial = direct_reduce ? at::Tensor{} : at::empty({batch, embed_dim}, projected.options());
  constexpr int kThreads = 256;
  const int elements = batch * embed_dim;
  const auto stream = at::hip::getCurrentHIPStream();
  if (direct_reduce) {
    hipError_t error = hipMemsetAsync(
        grad_decay_bias.data_ptr<float>(),
        0,
        static_cast<std::size_t>(decay_bias.numel()) * sizeof(float),
        stream);
    TORCH_CHECK(error == hipSuccess, "Mamba2 HIP grad_decay clear failed: ", hipGetErrorString(error));
    error = hipMemsetAsync(
        grad_skip.data_ptr<float>(),
        0,
        static_cast<std::size_t>(skip.numel()) * sizeof(float),
        stream);
    TORCH_CHECK(error == hipSuccess, "Mamba2 HIP grad_skip clear failed: ", hipGetErrorString(error));
    check_hip_launch();
  }
  if (has_reset) {
    if (direct_reduce) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(mamba2_scan_backward_kernel<true, true>),
          dim3(blocks_for(elements, kThreads)),
          dim3(kThreads),
          0,
          stream,
          grad_mixed.data_ptr<float>(),
          projected.data_ptr<float>(),
          decay_bias.data_ptr<float>(),
          skip.data_ptr<float>(),
          reset_mask.data_ptr<float>(),
          scan.data_ptr<float>(),
          grad_projected.data_ptr<float>(),
          grad_decay_bias.data_ptr<float>(),
          grad_skip.data_ptr<float>(),
          batch,
          sequence,
          embed_dim);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(mamba2_scan_backward_kernel<true, false>),
          dim3(blocks_for(elements, kThreads)),
          dim3(kThreads),
          0,
          stream,
          grad_mixed.data_ptr<float>(),
          projected.data_ptr<float>(),
          decay_bias.data_ptr<float>(),
          skip.data_ptr<float>(),
          reset_mask.data_ptr<float>(),
          scan.data_ptr<float>(),
          grad_projected.data_ptr<float>(),
          grad_decay_partial.data_ptr<float>(),
          grad_skip_partial.data_ptr<float>(),
          batch,
          sequence,
          embed_dim);
    }
  } else {
    if (direct_reduce) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(mamba2_scan_backward_kernel<false, true>),
          dim3(blocks_for(elements, kThreads)),
          dim3(kThreads),
          0,
          stream,
          grad_mixed.data_ptr<float>(),
          projected.data_ptr<float>(),
          decay_bias.data_ptr<float>(),
          skip.data_ptr<float>(),
          nullptr,
          scan.data_ptr<float>(),
          grad_projected.data_ptr<float>(),
          grad_decay_bias.data_ptr<float>(),
          grad_skip.data_ptr<float>(),
          batch,
          sequence,
          embed_dim);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(mamba2_scan_backward_kernel<false, false>),
          dim3(blocks_for(elements, kThreads)),
          dim3(kThreads),
          0,
          stream,
          grad_mixed.data_ptr<float>(),
          projected.data_ptr<float>(),
          decay_bias.data_ptr<float>(),
          skip.data_ptr<float>(),
          nullptr,
          scan.data_ptr<float>(),
          grad_projected.data_ptr<float>(),
          grad_decay_partial.data_ptr<float>(),
          grad_skip_partial.data_ptr<float>(),
          batch,
          sequence,
          embed_dim);
    }
  }
  check_hip_launch();
  if (direct_reduce) {
    return {grad_projected, grad_decay_bias, grad_skip};
  }
  hipLaunchKernelGGL(
      reduce_batch_dim_pair_kernel,
      dim3(embed_dim),
      dim3(kThreads),
      0,
      stream,
      grad_decay_partial.data_ptr<float>(),
      grad_skip_partial.data_ptr<float>(),
      grad_decay_bias.data_ptr<float>(),
      grad_skip.data_ptr<float>(),
      batch,
      embed_dim);
  check_hip_launch();
  return {grad_projected, grad_decay_bias, grad_skip};
}

std::tuple<at::Tensor, at::Tensor> mamba2_step_forward_hip(
    const at::Tensor& projected,
    const at::Tensor& previous_scan,
    const at::Tensor& decay_bias,
    const at::Tensor& skip) {
  set_hip_device_for(projected);
  const int batch = static_cast<int>(projected.size(0));
  const int embed_dim = static_cast<int>(projected.size(1) / 5);
  at::Tensor mixed = at::empty({batch, embed_dim}, projected.options());
  at::Tensor next_scan = at::empty({batch, embed_dim}, projected.options());
  constexpr int kThreads = 256;
  const int elements = batch * embed_dim;
  hipLaunchKernelGGL(
      mamba2_step_forward_kernel,
      dim3(blocks_for(elements, kThreads)),
      dim3(kThreads),
      0,
      at::hip::getCurrentHIPStream(),
      projected.data_ptr<float>(),
      previous_scan.data_ptr<float>(),
      decay_bias.data_ptr<float>(),
      skip.data_ptr<float>(),
      mixed.data_ptr<float>(),
      next_scan.data_ptr<float>(),
      batch,
      embed_dim);
  check_hip_launch();
  return {mixed, next_scan};
}

std::tuple<at::Tensor, at::Tensor> mamba2_causal_conv1d_silu_forward_hip(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& reset_mask) {
  set_hip_device_for(input);
  const int batch = static_cast<int>(input.size(0));
  const int sequence = static_cast<int>(input.size(1));
  const int embed_dim = static_cast<int>(input.size(2));
  at::Tensor output = at::empty_like(input);
  at::Tensor pre_activation = at::empty_like(input);
  constexpr int kThreads = 256;
  const int elements = batch * sequence * embed_dim;
  hipLaunchKernelGGL(
      causal_conv1d_silu_forward_kernel,
      dim3(blocks_for(elements, kThreads)),
      dim3(kThreads),
      0,
      at::hip::getCurrentHIPStream(),
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      reset_mask.data_ptr<float>(),
      output.data_ptr<float>(),
      pre_activation.data_ptr<float>(),
      batch,
      sequence,
      embed_dim);
  check_hip_launch();
  return {output, pre_activation};
}

std::vector<at::Tensor> mamba2_causal_conv1d_silu_backward_hip(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& reset_mask,
    const at::Tensor& pre_activation) {
  (void)bias;
  set_hip_device_for(input);
  const int batch = static_cast<int>(input.size(0));
  const int sequence = static_cast<int>(input.size(1));
  const int embed_dim = static_cast<int>(input.size(2));
  at::Tensor grad_input = at::empty_like(input);
  at::Tensor grad_weight = at::empty_like(weight);
  at::Tensor grad_bias = at::empty_like(bias);
  constexpr int kThreads = 256;
  const int elements = batch * sequence * embed_dim;
  const auto stream = at::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(
      causal_conv1d_silu_backward_input_kernel,
      dim3(blocks_for(elements, kThreads)),
      dim3(kThreads),
      0,
      stream,
      grad_output.data_ptr<float>(),
      pre_activation.data_ptr<float>(),
      weight.data_ptr<float>(),
      reset_mask.data_ptr<float>(),
      grad_input.data_ptr<float>(),
      batch,
      sequence,
      embed_dim);
  check_hip_launch();
  hipLaunchKernelGGL(
      causal_conv1d_silu_backward_params_kernel,
      dim3(embed_dim),
      dim3(kThreads),
      0,
      stream,
      grad_output.data_ptr<float>(),
      input.data_ptr<float>(),
      pre_activation.data_ptr<float>(),
      reset_mask.data_ptr<float>(),
      grad_weight.data_ptr<float>(),
      grad_bias.data_ptr<float>(),
      batch,
      sequence,
      embed_dim);
  check_hip_launch();
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace pulsar
