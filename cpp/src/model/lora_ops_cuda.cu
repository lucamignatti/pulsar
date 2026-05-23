#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace pulsar {
namespace {

template <typename scalar_t>
__global__ void eggroll_lora_perturb_kernel(
    const scalar_t* __restrict__ base_out,
    const scalar_t* __restrict__ x_A_stack,
    const scalar_t* __restrict__ B_stack,
    scalar_t* __restrict__ out,
    int population,
    int member_batch,
    int out_features,
    int rank,
    float perturb_scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = population * member_batch * out_features;
  if (index >= total) return;

  const int o = index % out_features;
  const int sample = index / out_features;
  const int p = sample / member_batch;

  float perturbation = 0.0F;
  const int x_a_stack_base = sample * rank;
  const int stack_b_base = (p * out_features + o) * rank;
  for (int r = 0; r < rank; ++r) {
      const float xas = static_cast<float>(x_A_stack[x_a_stack_base + r]);
      const float b_stack = static_cast<float>(B_stack[stack_b_base + r]);
      perturbation += xas * b_stack;
  }
  out[index] = static_cast<scalar_t>(static_cast<float>(base_out[index]) + perturb_scale * perturbation);
}

inline int blocks_for(int elements, int threads) {
  return (elements + threads - 1) / threads;
}

}  // namespace

at::Tensor eggroll_lora_perturb_cuda(
    const at::Tensor& base_out,
    const at::Tensor& x_A_stack,
    const at::Tensor& B_stack,
    float perturb_scale) {
  const c10::cuda::CUDAGuard guard(base_out.device());
  const int population = static_cast<int>(B_stack.size(0));
  const int member_batch = static_cast<int>(base_out.size(0) / population);
  const int out_features = static_cast<int>(base_out.size(1));
  const int rank = static_cast<int>(x_A_stack.size(1));
  at::Tensor out = at::empty_like(base_out);
  constexpr int kThreads = 256;
  const int total = static_cast<int>(base_out.numel());
  if (base_out.scalar_type() == at::kHalf) {
    eggroll_lora_perturb_kernel<at::Half><<<blocks_for(total, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        base_out.data_ptr<at::Half>(), x_A_stack.data_ptr<at::Half>(), B_stack.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(), population, member_batch, out_features, rank, perturb_scale);
  } else {
    eggroll_lora_perturb_kernel<float><<<blocks_for(total, kThreads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        base_out.data_ptr<float>(), x_A_stack.data_ptr<float>(), B_stack.data_ptr<float>(),
        out.data_ptr<float>(), population, member_batch, out_features, rank, perturb_scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace pulsar
