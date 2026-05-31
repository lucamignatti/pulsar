#pragma once

#ifdef PULSAR_HAS_TORCH

#include <string>
#include <vector>
#include <utility>
#include <torch/torch.h>

namespace pulsar {

struct CapturedGrad {
  torch::Tensor param;
  torch::Tensor grad;
};

struct GradientSanitizeResult {
  bool changed = false;
  std::string first_parameter;
};

torch::Tensor index_select_on_source_device(
    const torch::Tensor& tensor,
    int64_t dim,
    const torch::Tensor& indices);

void zero_existing_gradients(torch::nn::Module& module);

bool gradients_are_finite(const torch::nn::Module& module);

GradientSanitizeResult zero_nonfinite_gradients(torch::nn::Module& module);

torch::Tensor finite_or_zero(const torch::Tensor& tensor);

double clip_existing_gradients(torch::nn::Module& module, double max_norm);

void scale_existing_gradients(torch::nn::Module& module, double scale);

// Smooth L1 loss functions
torch::Tensor smooth_l1_value_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float delta);

torch::Tensor elementwise_smooth_l1_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float delta);

class ModuleRequiresGradGuard {
 public:
  explicit ModuleRequiresGradGuard(torch::nn::Module& module, bool requires_grad) {
    for (auto& param : module.parameters()) {
      previous_.push_back({param, param.requires_grad()});
      param.set_requires_grad(requires_grad);
    }
  }

  ~ModuleRequiresGradGuard() {
    for (auto& [param, requires_grad] : previous_) {
      param.set_requires_grad(requires_grad);
    }
  }

 private:
  std::vector<std::pair<torch::Tensor, bool>> previous_;
};

}  // namespace pulsar

#endif
