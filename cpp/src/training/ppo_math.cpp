#include "pulsar/training/ppo_math.hpp"

#ifdef PULSAR_HAS_TORCH

#include <ATen/autocast_mode.h>

#include <stdexcept>

namespace pulsar {
namespace {

class ScopedAutocast {
 public:
  ScopedAutocast(bool enabled, at::DeviceType device_type, at::ScalarType dtype)
      : enabled_(enabled), device_type_(device_type) {
    if (!enabled_) {
      return;
    }
    previous_enabled_ = at::autocast::is_autocast_enabled(device_type_);
    previous_dtype_ = at::autocast::get_autocast_dtype(device_type_);
    previous_cache_enabled_ = at::autocast::is_autocast_cache_enabled();
    at::autocast::set_autocast_enabled(device_type_, true);
    at::autocast::set_autocast_dtype(device_type_, dtype);
    at::autocast::set_autocast_cache_enabled(true);
    at::autocast::increment_nesting();
  }

  ~ScopedAutocast() {
    if (!enabled_) {
      return;
    }
    at::autocast::decrement_nesting();
    at::autocast::set_autocast_enabled(device_type_, previous_enabled_);
    at::autocast::set_autocast_dtype(device_type_, previous_dtype_);
    at::autocast::set_autocast_cache_enabled(previous_cache_enabled_);
  }

 private:
  bool enabled_ = false;
  at::DeviceType device_type_ = at::kCPU;
  bool previous_enabled_ = false;
  bool previous_cache_enabled_ = false;
  at::ScalarType previous_dtype_ = at::kFloat;
};

}  // namespace

torch::Tensor apply_action_mask_to_logits(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  return logits.masked_fill(action_masks.logical_not(), -1.0e9);
}

torch::Tensor sample_masked_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs) {
  const torch::Tensor masked = apply_action_mask_to_logits(logits, action_masks);
  const torch::Tensor probs = torch::softmax(masked, -1);
  const torch::Tensor actions =
      deterministic ? probs.argmax(-1) : probs.multinomial(1).squeeze(-1);
  if (log_probs != nullptr) {
    *log_probs = torch::log_softmax(masked, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
  }
  return actions;
}

torch::Tensor masked_action_entropy(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  const torch::Tensor masked = apply_action_mask_to_logits(logits, action_masks);
  const torch::Tensor probs = torch::softmax(masked, -1);
  const torch::Tensor valid_counts = action_masks.to(torch::kFloat32).sum(-1).clamp_min(1.0F);
  return -(probs * torch::log(probs + 1.0e-8)).sum(-1) / valid_counts.log().clamp_min(1.0e-6);
}

torch::Tensor categorical_value_projection(const torch::Tensor& returns, float v_min, float v_max, int num_atoms) {
  const float delta_z = (v_max - v_min) / static_cast<float>(num_atoms - 1);
  const torch::Tensor clamped = returns.clamp(v_min, v_max);
  const torch::Tensor b = (clamped - v_min) / delta_z;
  const torch::Tensor lower = b.floor().to(torch::kLong).clamp(0, num_atoms - 1);
  const torch::Tensor upper = b.ceil().to(torch::kLong).clamp(0, num_atoms - 1);
  const torch::Tensor upper_prob = b - lower.to(torch::kFloat32);
  const torch::Tensor lower_prob = 1.0 - upper_prob;
  torch::Tensor target = torch::zeros(
      {returns.size(0), num_atoms},
      torch::TensorOptions().dtype(torch::kFloat32).device(returns.device()));
  target.scatter_add_(1, lower.unsqueeze(-1), lower_prob.unsqueeze(-1));
  target.scatter_add_(1, upper.unsqueeze(-1), upper_prob.unsqueeze(-1));
  return target;
}

torch::Tensor compute_confidence_weights(
    const SharedActorCritic& model,
    const PPOConfig& config,
    const torch::Tensor& value_logits) {
  if (!config.use_confidence_weighting) {
    return torch::ones(
        {value_logits.size(0)},
        torch::TensorOptions().dtype(torch::kFloat32).device(value_logits.device()));
  }

  torch::Tensor weights;
  if (config.confidence_weight_type == "variance") {
    weights = 1.0 / (model->value_variance(value_logits) + config.confidence_weight_delta);
  } else {
    weights = 1.0 / (model->value_entropy(value_logits) + config.confidence_weight_delta);
  }
  if (config.normalize_confidence_weights) {
    weights = weights / weights.mean().clamp_min(1.0e-6);
  }
  return weights.detach();
}

torch::Tensor compute_adaptive_epsilon(
    const SharedActorCritic& model,
    const PPOConfig& config,
    const torch::Tensor& value_logits) {
  if (!config.use_adaptive_epsilon) {
    return torch::full(
        {value_logits.size(0)},
        config.clip_range,
        torch::TensorOptions().dtype(torch::kFloat32).device(value_logits.device()));
  }
  const torch::Tensor epsilon =
      config.clip_range / (1.0 + config.adaptive_epsilon_beta * model->value_variance(value_logits));
  return torch::clamp(epsilon, config.epsilon_min, config.epsilon_max).detach();
}

void validate_precision_mode_or_throw(const PPOConfig::PrecisionConfig& precision, const torch::Device& device) {
  if (precision.mode == "fp32") {
    return;
  }
  if (precision.mode != "amp_bf16") {
    throw std::runtime_error("Unsupported ppo.precision.mode: " + precision.mode);
  }
  if (!device.is_cuda()) {
    throw std::runtime_error("ppo.precision.mode=amp_bf16 requires a CUDA/ROCm device.");
  }

  try {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor a = torch::randn({4, 4}, options);
    torch::Tensor b = torch::randn({4, 4}, options);
    ScopedAutocast autocast(true, device.type(), at::kBFloat16);
    (void)torch::matmul(a, b);
  } catch (const std::exception& exc) {
    throw std::runtime_error(std::string("Failed to enable BF16 autocast on the selected device: ") + exc.what());
  }
}

}  // namespace pulsar

#endif
