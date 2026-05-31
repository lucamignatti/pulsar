#include "pulsar/training/gradient_ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

torch::Tensor index_select_on_source_device(
    const torch::Tensor& tensor,
    int64_t dim,
    const torch::Tensor& indices) {
  torch::Tensor select_indices = indices;
  if (select_indices.scalar_type() != torch::kLong) {
    select_indices = select_indices.to(torch::kLong);
  }
  if (select_indices.device() != tensor.device()) {
    select_indices = select_indices.to(tensor.device());
  }
  return tensor.index_select(dim, select_indices);
}

void accumulate_gradients(torch::nn::Module& module, std::vector<CapturedGrad>& accumulated) {
  if (accumulated.empty()) {
    for (auto& p : module.parameters()) {
      accumulated.push_back({p, torch::Tensor{}});
    }
  }
  size_t i = 0;
  for (auto& p : module.parameters()) {
    if (i >= accumulated.size()) {
      accumulated.push_back({p, torch::Tensor{}});
    }
    if (p.grad().defined()) {
      if (accumulated[i].grad.defined()) {
        accumulated[i].grad.add_(p.grad().detach());
      } else {
        accumulated[i].grad = p.grad().detach().clone();
      }
    }
    ++i;
  }
}

void reduce_captured_gradients(
    torch::nn::Module& module,
    std::vector<CapturedGrad>& dst,
    const std::vector<CapturedGrad>& src,
    const torch::Device& device) {
  if (src.empty()) {
    return;
  }
  if (dst.empty()) {
    for (auto& p : module.parameters()) {
      dst.push_back({p, torch::Tensor{}});
    }
  }
  for (size_t i = 0; i < src.size() && i < dst.size(); ++i) {
    if (!src[i].grad.defined()) {
      continue;
    }
    const torch::Tensor grad = src[i].grad.to(device);
    if (dst[i].grad.defined()) {
      dst[i].grad.add_(grad);
    } else {
      dst[i].grad = grad.clone();
    }
  }
}

void zero_existing_gradients(torch::nn::Module& module) {
  for (auto& p : module.parameters()) {
    torch::Tensor grad = p.mutable_grad();
    if (grad.defined()) {
      grad.zero_();
    }
  }
}

bool gradients_are_finite(const torch::nn::Module& module) {
  for (const auto& p : module.parameters()) {
    const torch::Tensor grad = p.grad();
    if (grad.defined() && !torch::isfinite(grad).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

GradientSanitizeResult zero_nonfinite_gradients(torch::nn::Module& module) {
  GradientSanitizeResult result;
  for (auto& item : module.named_parameters(true)) {
    torch::Tensor grad = item.value().mutable_grad();
    if (!grad.defined()) {
      continue;
    }
    const torch::Tensor finite = torch::isfinite(grad);
    if (!finite.all().item<bool>()) {
      if (!result.changed) {
        result.first_parameter = item.key();
      }
      grad.masked_fill_(finite.logical_not(), 0.0);
      result.changed = true;
    }
  }
  return result;
}

torch::Tensor finite_or_zero(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  return torch::where(torch::isfinite(tensor), tensor, torch::zeros_like(tensor));
}

double clip_existing_gradients(torch::nn::Module& module, double max_norm) {
  std::vector<torch::Tensor> grads;
  for (auto& p : module.parameters()) {
    const torch::Tensor grad = p.grad();
    if (grad.defined()) {
      grads.push_back(grad.detach().reshape({-1}));
    }
  }
  if (grads.empty()) {
    return 0.0;
  }

  torch::Tensor all_grads = torch::cat(grads);
  double max_abs = all_grads.abs().max().item<double>();
  if (!std::isfinite(max_abs)) {
    return max_abs;
  }
  if (max_abs == 0.0) {
    return 0.0;
  }

  torch::Tensor scaled = all_grads.to(torch::kFloat32) / max_abs;
  double param_scaled_sq = scaled.square().sum().item<double>();
  if (!std::isfinite(param_scaled_sq)) {
    return param_scaled_sq;
  }
  const double total_norm = max_abs * std::sqrt(param_scaled_sq);

  if (!std::isfinite(total_norm) || max_norm <= 0.0 || total_norm <= max_norm) {
    return total_norm;
  }

  const double scale = max_norm / (total_norm + 1.0e-6);
  for (auto& p : module.parameters()) {
    torch::Tensor grad = p.mutable_grad();
    if (grad.defined()) {
      grad.mul_(scale);
    }
  }
  return total_norm;
}

bool captured_group_has_grad(const std::vector<CapturedGrad>& group) {
  for (const auto& captured : group) {
    if (captured.grad.defined()) {
      return true;
    }
  }
  return false;
}

bool captured_group_gradients_are_finite(const std::vector<CapturedGrad>& group) {
  std::vector<torch::Tensor> grads;
  grads.reserve(group.size());
  for (const auto& captured : group) {
    if (captured.grad.defined()) {
      grads.push_back(captured.grad.reshape({-1}));
    }
  }
  if (grads.empty()) {
    return true;
  }
  return torch::isfinite(torch::cat(grads)).all().item<bool>();
}

void scale_existing_gradients(torch::nn::Module& module, double scale) {
  if (scale == 1.0) {
    return;
  }
  for (auto& p : module.parameters()) {
    torch::Tensor grad = p.mutable_grad();
    if (grad.defined()) {
      grad.mul_(scale);
    }
  }
}


torch::Tensor smooth_l1_value_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float delta) {
  if (delta <= 0.0F) {
    return torch::mse_loss(prediction, target, torch::Reduction::Mean);
  }
  const torch::Tensor error = prediction - target;
  const torch::Tensor abs_error = error.abs();
  const torch::Tensor delta_tensor = torch::full_like(abs_error, delta);
  const torch::Tensor quadratic = 0.5F * error.square() / delta;
  const torch::Tensor linear = abs_error - 0.5F * delta;
  return torch::where(abs_error < delta_tensor, quadratic, linear).mean();
}

torch::Tensor elementwise_smooth_l1_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float delta) {
  if (delta <= 0.0F) {
    return (prediction - target).square();
  }
  const torch::Tensor error = prediction - target;
  const torch::Tensor abs_error = error.abs();
  const torch::Tensor delta_tensor = torch::full_like(abs_error, delta);
  const torch::Tensor quadratic = 0.5F * error.square() / delta;
  const torch::Tensor linear = abs_error - 0.5F * delta;
  return torch::where(abs_error < delta_tensor, quadratic, linear);
}

}  // namespace pulsar

#endif
