#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>
#include <ATen/Context.h>

namespace pulsar {

inline void configure_cuda_runtime(const torch::Device& device) {
  if (!device.is_cuda()) {
    return;
  }
  at::globalContext().setAllowTF32CuBLAS(true);
  at::globalContext().setAllowTF32CuDNN(true);
}

}  // namespace pulsar

#endif
