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

    double grad_sq_sum = 0.0;
    for (auto& group : param_groups()) {
      for (auto& param : group.params()) {
        if (param.grad().defined()) {
          grad_sq_sum += param.grad().detach().to(torch::kFloat32).square().sum().cpu().item<double>();
        }
      }
    }

    const double grad_norm = std::sqrt(grad_sq_sum);
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
