#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

namespace pulsar {
namespace {

__global__ void eggroll_lora_perturb_kernel(
    const float* __restrict__ base_out,
    const float* __restrict__ x_A_stack,
    const float* __restrict__ B_stack,
    float* __restrict__ out,
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
    const float xas = x_A_stack[x_a_stack_base + r];
    const float b_stack = B_stack[stack_b_base + r];
    perturbation += xas * b_stack;
  }
  out[index] = base_out[index] + perturb_scale * perturbation;
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
      TORCH_CHECK(error == hipSuccess, "LoRA HIP set device failed: ", hipGetErrorString(error));
      current_device = device_index;
    }
  }
}

inline void check_hip_launch() {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "LoRA HIP kernel launch failed: ", hipGetErrorString(error));
}

}  // namespace

at::Tensor eggroll_lora_perturb_hip(
    const at::Tensor& base_out,
    const at::Tensor& x_A_stack,
    const at::Tensor& B_stack,
    float perturb_scale) {
  set_hip_device_for(base_out);
  const int population = static_cast<int>(B_stack.size(0));
  const int member_batch = static_cast<int>(base_out.size(0) / population);
  const int out_features = static_cast<int>(base_out.size(1));
  const int rank = static_cast<int>(x_A.size(1));
  at::Tensor out = at::empty_like(base_out);
  constexpr int kThreads = 256;
  const int total = static_cast<int>(base_out.numel());
  eggroll_lora_perturb_kernel<<<blocks_for(total, kThreads), kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      base_out.data_ptr<float>(),
      x_A_stack.data_ptr<float>(),
      B_stack.data_ptr<float>(),
      out.data_ptr<float>(),
      population,
      member_batch,
      out_features,
      rank,
      perturb_scale);
  check_hip_launch();
  return out;
}

}  // namespace pulsar
