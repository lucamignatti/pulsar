#pragma once

#ifdef PULSAR_HAS_TORCH

#include <tuple>

#include <torch/torch.h>

namespace pulsar {

[[nodiscard]] bool mamba2_cuda_kernels_available();

torch::Tensor mamba2_scan_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask = {});

std::tuple<torch::Tensor, torch::Tensor> mamba2_step_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& previous_scan,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip);

}  // namespace pulsar

#endif  // PULSAR_HAS_TORCH
