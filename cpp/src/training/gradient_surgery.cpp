#include "pulsar/training/gradient_surgery.hpp"

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

GradientSanitizeResult zero_nonfinite_parameters(torch::nn::Module& module) {
  GradientSanitizeResult result;
  for (auto& item : module.named_parameters(true)) {
    torch::Tensor param = item.value();
    if (!param.defined()) {
      continue;
    }
    const torch::Tensor finite = torch::isfinite(param);
    if (!finite.all().item<bool>()) {
      if (!result.changed) {
        result.first_parameter = item.key();
      }
      torch::NoGradGuard no_grad;
      const bool is_weight =
          item.key().size() >= std::string(".weight").size() &&
          item.key().compare(
              item.key().size() - std::string(".weight").size(),
              std::string(".weight").size(),
              ".weight") == 0;
      const bool default_to_one = item.key().find("norm") != std::string::npos && is_weight;
      const torch::Tensor replacement = default_to_one ? torch::ones_like(param) : torch::zeros_like(param);
      param.copy_(torch::where(finite, param, replacement));
      result.changed = true;
    }
  }
  return result;
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
      grad.div_(scale);
    }
  }
}

void apply_pcgrad_multi(std::vector<std::vector<CapturedGrad>>& groups) {
  if (groups.size() < 2) return;

  std::vector<torch::Tensor> flats;
  std::vector<std::pair<size_t, int64_t>> layout;
  std::vector<bool> active_params(groups[0].size(), false);
  flats.reserve(groups.size());
  layout.reserve(groups[0].size());

  for (size_t i = 0; i < groups[0].size(); ++i) {
    layout.push_back({i, groups[0][i].param.numel()});
  }
  std::vector<torch::Tensor> zero_parts(groups[0].size());
  for (auto& group : groups) {
    std::vector<torch::Tensor> parts;
    parts.reserve(group.size());
    for (size_t i = 0; i < groups[0].size(); ++i) {
      if (group[i].grad.defined()) {
        active_params[i] = true;
        parts.push_back(group[i].grad.view({-1}));
      } else {
        if (!zero_parts[i].defined()) {
          zero_parts[i] = torch::zeros({group[i].param.numel()}, group[i].param.options());
        }
        parts.push_back(zero_parts[i]);
      }
    }
    flats.push_back(torch::cat(parts, 0));
  }

  const std::vector<torch::Tensor> orig_flats = flats;
  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = 0; j < groups.size(); ++j) {
      if (i == j) continue;
      const torch::Tensor dot = flats[i].dot(orig_flats[j]);
      if (!torch::isfinite(dot).item<bool>()) {
        continue;
      }
      if (dot.item<float>() < 0.0F) {
        const torch::Tensor norm_j_sq = orig_flats[j].dot(orig_flats[j]).clamp_min(1.0e-12F);
        if (!torch::isfinite(norm_j_sq).item<bool>()) {
          continue;
        }
        const torch::Tensor coeff = dot / norm_j_sq;
        flats[i] = flats[i] - coeff * orig_flats[j];
      }
    }
  }

  for (size_t g = 0; g < groups.size(); ++g) {
    int64_t offset = 0;
    for (const auto& [param_idx, sz] : layout) {
      if (active_params[param_idx]) {
        groups[g][param_idx].grad =
            flats[g].slice(0, offset, offset + sz).view(groups[g][param_idx].param.sizes()).clone();
      } else {
        groups[g][param_idx].grad = torch::Tensor{};
      }
      offset += sz;
    }
  }
}

void reduce_captured_grad_groups(
    std::vector<std::vector<CapturedGrad>>& primary_groups,
    const std::vector<std::vector<std::vector<CapturedGrad>>>& replica_groups_list,
    const torch::Device& primary_device) {
  for (const auto& replica_groups : replica_groups_list) {
    if (replica_groups.empty()) continue;
    for (size_t g = 0; g < replica_groups.size(); ++g) {
      if (g >= primary_groups.size()) break;
      for (size_t p = 0; p < replica_groups[g].size(); ++p) {
        if (p >= primary_groups[g].size()) break;
        if (replica_groups[g][p].grad.defined()) {
          if (primary_groups[g][p].grad.defined()) {
            primary_groups[g][p].grad.add_(replica_groups[g][p].grad.to(primary_device));
          } else {
            primary_groups[g][p].grad = replica_groups[g][p].grad.to(primary_device);
          }
        }
      }
    }
  }
}

void reduce_gradients_from_replicas(
    VRPOActor& primary,
    const std::vector<VRPOActor>& replicas) {
  if (!primary) return;
  auto primary_params = primary->named_parameters(true);
  if (primary_params.size() == 0) return;
  const torch::Device primary_device = primary->parameters().front().device();
  for (const auto& replica : replicas) {
    if (!replica) continue;
    auto replica_params = replica->named_parameters(true);
    for (const auto& item : replica_params) {
      torch::Tensor* primary_tensor = primary_params.find(item.key());
      if (primary_tensor == nullptr) continue;
      torch::Tensor replica_grad = item.value().mutable_grad();
      if (!replica_grad.defined()) continue;
      torch::Tensor primary_grad = primary_tensor->mutable_grad();
      if (primary_grad.defined()) {
        primary_grad.add_(replica_grad.to(primary_device));
      } else {
        primary_tensor->mutable_grad() = replica_grad.to(primary_device);
      }
    }
  }
}

void sync_actor_to_replicas(
    const VRPOActor& primary,
    std::vector<VRPOActor>& replicas) {
  if (!primary) return;
  for (auto& replica : replicas) {
    if (replica) {
      copy_vrpo_actor_tensors_to(primary, replica, replica->parameters().front().device());
    }
  }
}

torch::Tensor policy_goal_values_like(const torch::Tensor& obs, int goal_dim) {
  const auto options = obs.options().dtype(torch::kFloat32);
  if (obs.dim() == 3) {
    return torch::zeros({obs.size(0), obs.size(1), goal_dim}, options);
  }
  return torch::zeros({obs.size(0), goal_dim}, options);
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
