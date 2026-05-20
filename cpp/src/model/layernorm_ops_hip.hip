#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

namespace pulsar {
namespace {

constexpr float kLayerNormEpsDefault = 1.0e-5F;

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

inline void set_hip_device_for(const at::Tensor& tensor) {
  const int device_index = tensor.get_device();
  if (device_index >= 0) {
    const hipError_t error = hipSetDevice(device_index);
    TORCH_CHECK(error == hipSuccess, "layernorm HIP set device failed: ", hipGetErrorString(error));
  }
}

inline void check_hip_launch() {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "layernorm HIP kernel launch failed: ", hipGetErrorString(error));
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
  __shared__ float smem_sum[Threads];
  __shared__ float smem_sumsq[Threads];

  const int row = blockIdx.x;
  if (row >= N) return;

  const float* x = input + row * D;
  float* y = output + row * D;

  float sum = 0.0F;
  float sumsq = 0.0F;
  for (int i = threadIdx.x; i < D; i += Threads) {
    const float value = x[i];
    sum += value;
    sumsq += value * value;
  }
  smem_sum[threadIdx.x] = sum;
  smem_sumsq[threadIdx.x] = sumsq;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem_sum[threadIdx.x] += smem_sum[threadIdx.x + stride];
      smem_sumsq[threadIdx.x] += smem_sumsq[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_d = 1.0F / static_cast<float>(D);
  const float mean = smem_sum[0] * inv_d;
  const float variance = fmaxf(smem_sumsq[0] * inv_d - mean * mean, 0.0F);
  const float inv_std = rsqrtf(variance + eps);

  if (threadIdx.x == 0 && saved_mean != nullptr && saved_inv_std != nullptr) {
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
    int N,
    int D) {
  __shared__ float smem_sum1[Threads + 1];
  __shared__ float smem_sum2[Threads + 1];

  const int row = blockIdx.x;
  if (row >= N) return;

  const float* dy = grad_output + row * D;
  const float* x = input + row * D;
  float* dx = grad_input + row * D;
  const float mean = saved_mean[row];
  const float inv_std = saved_inv_std[row];

  float sum_dxhat = 0.0F;
  float sum_dxhat_xhat = 0.0F;

  if (weight != nullptr) {
    for (int i = threadIdx.x; i < D; i += Threads) {
      const float x_hat = (x[i] - mean) * inv_std;
      const float dx_hat = dy[i] * weight[i];
      sum_dxhat += dx_hat;
      sum_dxhat_xhat += dx_hat * x_hat;
    }
  } else {
    for (int i = threadIdx.x; i < D; i += Threads) {
      const float x_hat = (x[i] - mean) * inv_std;
      const float dx_hat = dy[i];
      sum_dxhat += dx_hat;
      sum_dxhat_xhat += dx_hat * x_hat;
    }
  }

  smem_sum1[threadIdx.x] = sum_dxhat;
  smem_sum2[threadIdx.x] = sum_dxhat_xhat;
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
}

template <int Cols, int Rows, int RowsPerBlock>
__global__ void layernorm_backward_params_stage1_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ saved_mean,
    const float* __restrict__ saved_inv_std,
    float* __restrict__ partial_weight,
    float* __restrict__ partial_bias,
    int N,
    int D,
    int chunks) {
  __shared__ float weight_sums[Rows][Cols];
  __shared__ float bias_sums[Rows][Cols];

  const int col_lane = threadIdx.x;
  const int row_lane = threadIdx.y;
  const int d = blockIdx.x * Cols + col_lane;
  const int chunk = blockIdx.y;
  const bool valid_col = chunk < chunks && d < D;

  const int row_begin = chunk * RowsPerBlock;
  const int row_end = row_begin + RowsPerBlock < N ? row_begin + RowsPerBlock : N;
  float w_sum = 0.0F;
  float b_sum = 0.0F;
  if (valid_col) {
    for (int row = row_begin + row_lane; row < row_end; row += Rows) {
      const int offset = row * D + d;
      const float dy = grad_output[offset];
      b_sum += dy;
      if (partial_weight != nullptr) {
        const float x_hat = (input[offset] - saved_mean[row]) * saved_inv_std[row];
        w_sum += dy * x_hat;
      }
    }
  }

  weight_sums[row_lane][col_lane] = w_sum;
  bias_sums[row_lane][col_lane] = b_sum;
  __syncthreads();

  for (int stride = Rows / 2; stride > 0; stride >>= 1) {
    if (row_lane < stride) {
      weight_sums[row_lane][col_lane] += weight_sums[row_lane + stride][col_lane];
      bias_sums[row_lane][col_lane] += bias_sums[row_lane + stride][col_lane];
    }
    __syncthreads();
  }

  if (row_lane == 0 && valid_col) {
    const int partial_offset = chunk * D + d;
    if (partial_weight != nullptr) partial_weight[partial_offset] = weight_sums[0][col_lane];
    if (partial_bias != nullptr) partial_bias[partial_offset] = bias_sums[0][col_lane];
  }
}

__global__ void layernorm_backward_params_stage2_kernel(
    const float* __restrict__ partial_weight,
    const float* __restrict__ partial_bias,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int chunks,
    int D) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;

  float w_sum = 0.0F;
  float b_sum = 0.0F;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int offset = chunk * D + d;
    if (partial_weight != nullptr) w_sum += partial_weight[offset];
    if (partial_bias != nullptr) b_sum += partial_bias[offset];
  }
  if (grad_weight) grad_weight[d] = w_sum;
  if (grad_bias) grad_bias[d] = b_sum;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_layer_norm_forward_hip(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    float eps) {
  set_hip_device_for(input);

  const int N = input.numel() / input.size(-1);
  const int D = static_cast<int>(input.size(-1));
  const auto options = input.options();

  at::Tensor output = at::empty_like(input);
  at::Tensor saved_mean = at::empty({N}, options);
  at::Tensor saved_inv_std = at::empty({N}, options);

  const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
  const float* b_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

  constexpr int kThreads = 128;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(layernorm_forward_kernel<kThreads>),
      dim3(N),
      dim3(kThreads),
      0,
      at::hip::getCurrentHIPStream(),
      input.data_ptr<float>(),
      w_ptr,
      b_ptr,
      output.data_ptr<float>(),
      saved_mean.data_ptr<float>(),
      saved_inv_std.data_ptr<float>(),
      N, D, eps);
  check_hip_launch();
  return {output, saved_mean, saved_inv_std};
}

at::Tensor fused_layer_norm_inference_hip(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    float eps) {
  set_hip_device_for(input);

  const int N = input.numel() / input.size(-1);
  const int D = static_cast<int>(input.size(-1));
  at::Tensor output = at::empty_like(input);

  const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
  const float* b_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

  constexpr int kThreads = 128;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(layernorm_forward_kernel<kThreads>),
      dim3(N),
      dim3(kThreads),
      0,
      at::hip::getCurrentHIPStream(),
      input.data_ptr<float>(),
      w_ptr,
      b_ptr,
      output.data_ptr<float>(),
      nullptr,
      nullptr,
      N, D, eps);
  check_hip_launch();
  return output;
}

std::vector<at::Tensor> fused_layer_norm_backward_hip(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& saved_mean,
    const at::Tensor& saved_inv_std) {
  set_hip_device_for(grad_output);

  const int N = grad_output.numel() / grad_output.size(-1);
  const int D = static_cast<int>(grad_output.size(-1));
  const auto options = grad_output.options();
  const auto stream = at::hip::getCurrentHIPStream();

  const bool has_weight = weight.defined();
  const bool has_bias = bias.defined();
  const float* w_ptr = has_weight ? weight.data_ptr<float>() : nullptr;

  at::Tensor grad_input = at::empty_like(grad_output);
  at::Tensor grad_weight = has_weight ? at::empty_like(weight) : at::Tensor{};
  at::Tensor grad_bias = has_bias ? at::empty_like(bias) : at::Tensor{};

  const bool needs_params = has_weight || has_bias;

  constexpr int kThreads = 128;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(layernorm_backward_kernel<kThreads>),
      dim3(N),
      dim3(kThreads),
      0,
      stream,
      grad_output.data_ptr<float>(),
      input.data_ptr<float>(),
      w_ptr,
      saved_mean.data_ptr<float>(),
      saved_inv_std.data_ptr<float>(),
      grad_input.data_ptr<float>(),
      N, D);
  check_hip_launch();

  if (needs_params) {
    constexpr int kParamCols = 32;
    constexpr int kParamRows = 8;
    constexpr int kParamRowsPerBlock = 256;
    const int chunks = blocks_for(N, kParamRowsPerBlock);
    at::Tensor partial_grad_weight = has_weight ? at::empty({chunks, D}, options) : at::Tensor{};
    at::Tensor partial_grad_bias = has_bias ? at::empty({chunks, D}, options) : at::Tensor{};
    float* pw_ptr = has_weight ? partial_grad_weight.data_ptr<float>() : nullptr;
    float* pb_ptr = has_bias ? partial_grad_bias.data_ptr<float>() : nullptr;
    const dim3 stage1_grid(blocks_for(D, kParamCols), chunks);
    const dim3 stage1_block(kParamCols, kParamRows);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(layernorm_backward_params_stage1_kernel<kParamCols, kParamRows, kParamRowsPerBlock>),
        stage1_grid,
        stage1_block,
        0,
        stream,
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        saved_mean.data_ptr<float>(),
        saved_inv_std.data_ptr<float>(),
        pw_ptr,
        pb_ptr,
        N,
        D,
        chunks);
    check_hip_launch();

    constexpr int kReduceThreads = 256;
    const int reduce_blocks = blocks_for(D, kReduceThreads);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(layernorm_backward_params_stage2_kernel),
        dim3(reduce_blocks),
        dim3(kReduceThreads),
        0,
        stream,
        pw_ptr,
        pb_ptr,
        has_weight ? grad_weight.data_ptr<float>() : nullptr,
        has_bias ? grad_bias.data_ptr<float>() : nullptr,
        chunks,
        D);
    check_hip_launch();
  }

  return {grad_input, grad_weight, grad_bias};
}

}  // namespace pulsar
