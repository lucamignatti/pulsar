#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace pulsar {
namespace {

constexpr float kLayerNormEpsDefault = 1.0e-5F;

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

template <int Threads>
__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ saved_mean,
    float* __restrict__ saved_inv_std,
    int N,
    int D,
    float eps) {
  __shared__ float smem[Threads + 1];

  const int row = blockIdx.x;
  if (row >= N) return;

  const float* x = input + row * D;
  float* y = output + row * D;

  float sum = 0.0F;
  for (int i = threadIdx.x; i < D; i += Threads) {
    sum += x[i];
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = smem[0] / static_cast<float>(D);

  float var = 0.0F;
  for (int i = threadIdx.x; i < D; i += Threads) {
    const float diff = x[i] - mean;
    var += diff * diff;
  }
  smem[threadIdx.x] = var;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_std = rsqrtf(smem[0] / static_cast<float>(D) + eps);

  if (threadIdx.x == 0) {
    saved_mean[row] = mean;
    saved_inv_std[row] = inv_std;
  }

  if (weight != nullptr && bias != nullptr) {
    for (int i = threadIdx.x; i < D; i += Threads) {
      y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
  } else if (weight != nullptr) {
    for (int i = threadIdx.x; i < D; i += Threads) {
      y[i] = (x[i] - mean) * inv_std * weight[i];
    }
  } else if (bias != nullptr) {
    for (int i = threadIdx.x; i < D; i += Threads) {
      y[i] = (x[i] - mean) * inv_std + bias[i];
    }
  } else {
    for (int i = threadIdx.x; i < D; i += Threads) {
      y[i] = (x[i] - mean) * inv_std;
    }
  }
}

template <int Threads>
__global__ void layernorm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ saved_mean,
    const float* __restrict__ saved_inv_std,
    float* __restrict__ grad_input,
    float* __restrict__ partial_grad_weight,
    float* __restrict__ partial_grad_bias,
    int N,
    int D) {
  __shared__ float smem_sum1[Threads + 1];
  __shared__ float smem_sum2[Threads + 1];
  __shared__ float smem_wsum[Threads];
  __shared__ float smem_bsum[Threads];

  const int row = blockIdx.x;
  if (row >= N) return;

  const float* dy = grad_output + row * D;
  const float* x = input + row * D;
  float* dx = grad_input + row * D;
  const float mean = saved_mean[row];
  const float inv_std = saved_inv_std[row];

  float sum_dxhat = 0.0F;
  float sum_dxhat_xhat = 0.0F;
  float local_wsum = 0.0F;
  float local_bsum = 0.0F;

  if (weight != nullptr) {
    for (int i = threadIdx.x; i < D; i += Threads) {
      const float x_hat = (x[i] - mean) * inv_std;
      const float dx_hat = dy[i] * weight[i];
      sum_dxhat += dx_hat;
      sum_dxhat_xhat += dx_hat * x_hat;
      local_wsum += dy[i] * x_hat;
      local_bsum += dy[i];
    }
  } else {
    for (int i = threadIdx.x; i < D; i += Threads) {
      const float x_hat = (x[i] - mean) * inv_std;
      const float dx_hat = dy[i];
      sum_dxhat += dx_hat;
      sum_dxhat_xhat += dx_hat * x_hat;
      local_bsum += dy[i];
    }
  }

  smem_sum1[threadIdx.x] = sum_dxhat;
  smem_sum2[threadIdx.x] = sum_dxhat_xhat;
  smem_wsum[threadIdx.x] = local_wsum;
  smem_bsum[threadIdx.x] = local_bsum;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem_sum1[threadIdx.x] += smem_sum1[threadIdx.x + stride];
      smem_sum2[threadIdx.x] += smem_sum2[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float row_sum_dxhat = smem_sum1[0];
  const float row_sum_dxhat_xhat = smem_sum2[0];

  if (weight != nullptr) {
    for (int i = threadIdx.x; i < D; i += Threads) {
      const float x_hat = (x[i] - mean) * inv_std;
      const float dx_hat = dy[i] * weight[i];
      dx[i] = inv_std * (dx_hat - row_sum_dxhat / static_cast<float>(D) - x_hat * row_sum_dxhat_xhat / static_cast<float>(D));
    }
  } else {
    for (int i = threadIdx.x; i < D; i += Threads) {
      const float x_hat = (x[i] - mean) * inv_std;
      dx[i] = inv_std * (dy[i] - row_sum_dxhat / static_cast<float>(D) - x_hat * row_sum_dxhat_xhat / static_cast<float>(D));
    }
  }

  if (weight != nullptr) {
    for (int stride = Threads / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        smem_wsum[threadIdx.x] += smem_wsum[threadIdx.x + stride];
        smem_bsum[threadIdx.x] += smem_bsum[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      partial_grad_weight[row] = smem_wsum[0];
      partial_grad_bias[row] = smem_bsum[0];
    }
  } else {
    for (int stride = Threads / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        smem_bsum[threadIdx.x] += smem_bsum[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      partial_grad_bias[row] = smem_bsum[0];
    }
  }
}

__global__ void layernorm_backward_params_reduce_kernel(
    const float* __restrict__ partial_weight,
    const float* __restrict__ partial_bias,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int N,
    int D) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;

  float w_sum = 0.0F;
  float b_sum = 0.0F;
  for (int n = 0; n < N; ++n) {
    w_sum += partial_weight[n * D + d];
    b_sum += partial_bias[n * D + d];
  }
  if (grad_weight) grad_weight[d] = w_sum;
  if (grad_bias) grad_bias[d] = b_sum;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_layer_norm_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    float eps) {
  TORCH_CHECK(input.is_cuda(), "fused_layer_norm CUDA kernel requires CUDA input tensor.");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "fused_layer_norm CUDA kernel supports float32 input.");
  const c10::cuda::CUDAGuard device_guard(input.device());

  const int N = input.numel() / input.size(-1);
  const int D = static_cast<int>(input.size(-1));
  const auto options = input.options();

  at::Tensor output = at::empty_like(input);
  at::Tensor saved_mean = at::empty({N}, options);
  at::Tensor saved_inv_std = at::empty({N}, options);

  const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
  const float* b_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

  constexpr int kThreads = 128;
  const auto stream = at::cuda::getCurrentCUDAStream();
  layernorm_forward_kernel<kThreads><<<N, kThreads, 0, stream>>>(
      input.data_ptr<float>(),
      w_ptr,
      b_ptr,
      output.data_ptr<float>(),
      saved_mean.data_ptr<float>(),
      saved_inv_std.data_ptr<float>(),
      N, D, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, saved_mean, saved_inv_std};
}

std::vector<at::Tensor> fused_layer_norm_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& saved_mean,
    const at::Tensor& saved_inv_std) {
  TORCH_CHECK(grad_output.is_cuda(), "fused_layer_norm CUDA kernel requires CUDA grad_output tensor.");
  TORCH_CHECK(grad_output.scalar_type() == at::kFloat, "fused_layer_norm CUDA kernel supports float32 grad_output.");
  const c10::cuda::CUDAGuard device_guard(grad_output.device());

  const int N = grad_output.numel() / grad_output.size(-1);
  const int D = static_cast<int>(grad_output.size(-1));
  const auto options = grad_output.options();
  const auto stream = at::cuda::getCurrentCUDAStream();

  const bool has_weight = weight.defined();
  const bool has_bias = bias.defined();
  const float* w_ptr = has_weight ? weight.data_ptr<float>() : nullptr;

  at::Tensor grad_input = at::empty_like(grad_output);
  at::Tensor grad_weight = has_weight ? at::empty_like(weight) : at::Tensor{};
  at::Tensor grad_bias = has_bias ? at::empty_like(bias) : at::Tensor{};

  const bool needs_params = has_weight || has_bias;
  at::Tensor partial_grad_weight = needs_params ? at::empty({N, D}, options) : at::Tensor{};
  at::Tensor partial_grad_bias = needs_params ? at::empty({N, D}, options) : at::Tensor{};

  float* pw_ptr = needs_params ? partial_grad_weight.data_ptr<float>() : nullptr;
  float* pb_ptr = needs_params ? partial_grad_bias.data_ptr<float>() : nullptr;

  constexpr int kThreads = 128;
  layernorm_backward_kernel<kThreads><<<N, kThreads, 0, stream>>>(
      grad_output.data_ptr<float>(),
      input.data_ptr<float>(),
      w_ptr,
      saved_mean.data_ptr<float>(),
      saved_inv_std.data_ptr<float>(),
      grad_input.data_ptr<float>(),
      pw_ptr,
      pb_ptr,
      N, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (needs_params) {
    constexpr int kReduceThreads = 256;
    const int reduce_blocks = blocks_for(D, kReduceThreads);
    layernorm_backward_params_reduce_kernel<<<reduce_blocks, kReduceThreads, 0, stream>>>(
        partial_grad_weight.data_ptr<float>(),
        partial_grad_bias.data_ptr<float>(),
        has_weight ? grad_weight.data_ptr<float>() : nullptr,
        has_bias ? grad_bias.data_ptr<float>() : nullptr,
        N, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {grad_input, grad_weight, grad_bias};
}

}  // namespace pulsar
