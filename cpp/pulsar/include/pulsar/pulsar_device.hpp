// Device selection for the Pulsar training system.
// Priority: MPS (Apple Silicon) > CUDA > CPU.
// No custom Metal/CUDA kernels are needed; PyTorch's built-in MPS backend is used.
#pragma once

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>

namespace pulsar {

inline torch::Device select_device() {
#if defined(__APPLE__)
  if (torch::mps::is_available()) {
    return torch::Device(torch::kMPS);
  }
#endif
  if (torch::cuda::is_available()) {
    return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

inline std::string device_name(const torch::Device& dev) {
  switch (dev.type()) {
    case torch::kMPS:  return "mps";
    case torch::kCUDA: return "cuda";
    default:           return "cpu";
  }
}

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
