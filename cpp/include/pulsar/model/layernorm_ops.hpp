#pragma once

#ifdef PULSAR_HAS_TORCH

#include <torch/torch.h>

namespace pulsar {

torch::Tensor fused_layer_norm(const torch::Tensor& input, const torch::nn::LayerNorm& norm);

}  // namespace pulsar

#endif  // PULSAR_HAS_TORCH
