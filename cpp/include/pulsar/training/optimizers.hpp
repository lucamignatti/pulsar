#pragma once

#ifdef PULSAR_HAS_TORCH

#include <cmath>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace pulsar {

class MagSGD : public torch::optim::SGD {
 public:
  MagSGD(std::vector<torch::Tensor> params, torch::optim::SGDOptions options)
      : torch::optim::SGD(std::move(params), std::move(options)) {}

  torch::Tensor step(LossClosure closure = nullptr) override {
    torch::NoGradGuard no_grad;
    torch::Tensor loss;
    if (closure != nullptr) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }

    // Compute global grad norm in one device-side op (no per-parameter sync).
    std::vector<torch::Tensor> grad_parts;
    for (auto& group : param_groups()) {
      for (auto& param : group.params()) {
        if (param.grad().defined()) {
          grad_parts.push_back(param.grad().detach().to(torch::kFloat32).reshape({-1}));
        }
      }
    }

    const double grad_norm = grad_parts.empty()
        ? 0.0
        : std::sqrt(torch::cat(grad_parts).square().sum().item<double>());

    if (std::isfinite(grad_norm) && grad_norm > 0.0) {
      for (auto& group : param_groups()) {
        for (auto& param : group.params()) {
          if (param.grad().defined()) {
            param.mutable_grad().div_(grad_norm);
          }
        }
      }
    }

    torch::optim::SGD::step(nullptr);
    return loss;
  }
};

}  // namespace pulsar

#endif
