#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

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
  if (index >= total) {
    return;
  }

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

template <bool HasReset>
__global__ void mamba2_scan_backward_kernel(
    const float* __restrict__ grad_mixed,
    const float* __restrict__ projected,
    const float* __restrict__ decay_bias,
    const float* __restrict__ skip,
    const float* __restrict__ reset,
    const float* __restrict__ scan,
    float* __restrict__ grad_projected,
    float* __restrict__ grad_decay_partial,
    float* __restrict__ grad_skip_partial,
    int batch,
    int sequence,
    int embed_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * embed_dim;
  if (index >= total) {
    return;
  }

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
    float grad_state = grad_mixed_pre * gate_c + grad_next_scan;
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
    const float retention_grad_active =
        (retention_unclamped > kRetentionMin && retention_unclamped < kRetentionMax) ? 1.0F : 0.0F;
    const float grad_retention_pre =
        grad_retention * retention_grad_active * retention_unclamped * (1.0F - retention_unclamped);
    grad_projected[proj_base + 4 * embed_dim] = grad_retention_pre;
    local_grad_decay += grad_retention_pre;
  }

  const int partial_index = b * embed_dim + d;
  grad_skip_partial[partial_index] = local_grad_skip;
  grad_decay_partial[partial_index] = local_grad_decay;
}

__global__ void reduce_batch_dim_pair_kernel(
    const float* __restrict__ left_partial,
    const float* __restrict__ right_partial,
    float* __restrict__ left_reduced,
    float* __restrict__ right_reduced,
    int batch,
    int embed_dim) {
  const int d = blockIdx.x;
  if (d >= embed_dim) {
    return;
  }
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
  if (index >= total) {
    return;
  }

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

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

void check_cuda_inputs(const at::Tensor& projected, const at::Tensor& decay_bias, const at::Tensor& skip) {
  TORCH_CHECK(projected.is_cuda(), "Mamba2 CUDA kernel requires CUDA projected tensor.");
  TORCH_CHECK(decay_bias.is_cuda(), "Mamba2 CUDA kernel requires CUDA decay_bias tensor.");
  TORCH_CHECK(skip.is_cuda(), "Mamba2 CUDA kernel requires CUDA skip tensor.");
  TORCH_CHECK(projected.scalar_type() == at::kFloat, "Mamba2 CUDA kernel currently supports float32 projected tensors.");
  TORCH_CHECK(decay_bias.scalar_type() == at::kFloat, "Mamba2 CUDA kernel currently supports float32 decay_bias tensors.");
  TORCH_CHECK(skip.scalar_type() == at::kFloat, "Mamba2 CUDA kernel currently supports float32 skip tensors.");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> mamba2_scan_forward_cuda(
    const at::Tensor& projected,
    const at::Tensor& decay_bias,
    const at::Tensor& skip,
    const at::Tensor& reset_mask) {
  check_cuda_inputs(projected, decay_bias, skip);
  const c10::cuda::CUDAGuard device_guard(projected.device());
  const int batch = static_cast<int>(projected.size(0));
  const int sequence = static_cast<int>(projected.size(1));
  const int embed_dim = static_cast<int>(projected.size(2) / 5);
  const bool has_reset = reset_mask.defined();
  if (has_reset) {
    TORCH_CHECK(reset_mask.is_cuda(), "Mamba2 CUDA kernel requires CUDA reset_mask tensor.");
    TORCH_CHECK(reset_mask.scalar_type() == at::kFloat, "Mamba2 CUDA kernel currently supports float32 reset_mask tensors.");
  }

  at::Tensor mixed = at::empty({batch, sequence, embed_dim}, projected.options());
  at::Tensor scan = at::empty({batch, sequence, embed_dim}, projected.options());
  constexpr int kThreads = 256;
  const int elements = batch * embed_dim;
  if (has_reset) {
    mamba2_scan_forward_kernel<true><<<blocks_for(elements, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    mamba2_scan_forward_kernel<false><<<blocks_for(elements, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {mixed, scan};
}

std::vector<at::Tensor> mamba2_scan_backward_cuda(
    const at::Tensor& grad_mixed,
    const at::Tensor& projected,
    const at::Tensor& decay_bias,
    const at::Tensor& skip,
    const at::Tensor& reset_mask,
    const at::Tensor& scan) {
  check_cuda_inputs(projected, decay_bias, skip);
  const c10::cuda::CUDAGuard device_guard(projected.device());
  const int batch = static_cast<int>(projected.size(0));
  const int sequence = static_cast<int>(projected.size(1));
  const int embed_dim = static_cast<int>(projected.size(2) / 5);
  const bool has_reset = reset_mask.defined();

  at::Tensor grad_projected = at::empty_like(projected);
  at::Tensor grad_decay_partial = at::empty({batch, embed_dim}, projected.options());
  at::Tensor grad_skip_partial = at::empty({batch, embed_dim}, projected.options());
  at::Tensor grad_decay_bias = at::empty_like(decay_bias);
  at::Tensor grad_skip = at::empty_like(skip);
  constexpr int kThreads = 256;
  const int elements = batch * embed_dim;
  if (has_reset) {
    mamba2_scan_backward_kernel<true><<<blocks_for(elements, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
  } else {
    mamba2_scan_backward_kernel<false><<<blocks_for(elements, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  reduce_batch_dim_pair_kernel<<<embed_dim, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      grad_decay_partial.data_ptr<float>(),
      grad_skip_partial.data_ptr<float>(),
      grad_decay_bias.data_ptr<float>(),
      grad_skip.data_ptr<float>(),
      batch,
      embed_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_projected, grad_decay_bias, grad_skip};
}

std::tuple<at::Tensor, at::Tensor> mamba2_step_forward_cuda(
    const at::Tensor& projected,
    const at::Tensor& previous_scan,
    const at::Tensor& decay_bias,
    const at::Tensor& skip) {
  check_cuda_inputs(projected, decay_bias, skip);
  TORCH_CHECK(previous_scan.is_cuda(), "Mamba2 CUDA kernel requires CUDA previous_scan tensor.");
  TORCH_CHECK(previous_scan.scalar_type() == at::kFloat, "Mamba2 CUDA kernel currently supports float32 previous_scan tensors.");
  const c10::cuda::CUDAGuard device_guard(projected.device());
  const int batch = static_cast<int>(projected.size(0));
  const int embed_dim = static_cast<int>(projected.size(1) / 5);
  at::Tensor mixed = at::empty({batch, embed_dim}, projected.options());
  at::Tensor next_scan = at::empty({batch, embed_dim}, projected.options());
  constexpr int kThreads = 256;
  const int elements = batch * embed_dim;
  mamba2_step_forward_kernel<<<blocks_for(elements, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      projected.data_ptr<float>(),
      previous_scan.data_ptr<float>(),
      decay_bias.data_ptr<float>(),
      skip.data_ptr<float>(),
      mixed.data_ptr<float>(),
      next_scan.data_ptr<float>(),
      batch,
      embed_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {mixed, next_scan};
}

}  // namespace pulsar
