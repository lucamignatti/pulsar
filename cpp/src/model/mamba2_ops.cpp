#include "pulsar/model/mamba2_ops.hpp"

#ifdef PULSAR_HAS_TORCH

#include <stdexcept>
#include <vector>

#include <torch/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

namespace pulsar {
namespace {

torch::Tensor fallback_mamba2_scan_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask) {
  const auto batch = projected.size(0);
  const auto sequence = projected.size(1);
  const auto embed_dim = projected.size(2) / 5;
  const auto chunks = projected.chunk(5, -1);
  const torch::Tensor x = torch::silu(chunks[0]);
  const torch::Tensor b = torch::sigmoid(chunks[1]);
  const torch::Tensor c = torch::sigmoid(chunks[2]);
  const torch::Tensor z = torch::silu(chunks[3]);
  const torch::Tensor retention =
      torch::sigmoid(chunks[4] + decay_bias.view({1, 1, embed_dim})).clamp(0.01, 0.9999);
  const torch::Tensor recurrent_input = b * x;
  const torch::Tensor reset = reset_mask.defined()
      ? reset_mask.to(projected.device()).to(projected.scalar_type())
      : torch::Tensor{};

  torch::Tensor state = torch::zeros({batch, embed_dim}, projected.options());
  std::vector<torch::Tensor> states;
  states.reserve(static_cast<std::size_t>(sequence));
  for (int64_t t = 0; t < sequence; ++t) {
    if (reset.defined()) {
      const torch::Tensor keep = (1.0F - reset.select(1, t)).view({batch, 1});
      state = state * keep;
    }
    state = retention.select(1, t) * state + recurrent_input.select(1, t);
    states.push_back(state);
  }
  const torch::Tensor scanned = torch::stack(states, 1);
  return (c * scanned + skip.view({1, 1, embed_dim}) * x) * z;
}

std::tuple<torch::Tensor, torch::Tensor> fallback_mamba2_step_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& previous_scan,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip) {
  const auto embed_dim = projected.size(1) / 5;
  const auto chunks = projected.chunk(5, -1);
  const torch::Tensor x = torch::silu(chunks[0]);
  const torch::Tensor b = torch::sigmoid(chunks[1]);
  const torch::Tensor c = torch::sigmoid(chunks[2]);
  const torch::Tensor z = torch::silu(chunks[3]);
  const torch::Tensor retention =
      torch::sigmoid(chunks[4] + decay_bias.view({1, embed_dim})).clamp(0.01, 0.9999);
  const torch::Tensor scan = retention * previous_scan + b * x;
  const torch::Tensor mixed = (c * scan + skip.view({1, embed_dim}) * x) * z;
  return {mixed, scan};
}

void validate_scan_inputs(
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask) {
  if (!projected.defined() || projected.dim() != 3) {
    throw std::invalid_argument("mamba2_scan_mixed requires projected with shape [batch, sequence, 5 * dim].");
  }
  if (projected.size(2) % 5 != 0) {
    throw std::invalid_argument("mamba2_scan_mixed projected last dimension must be divisible by 5.");
  }
  const auto embed_dim = projected.size(2) / 5;
  if (!decay_bias.defined() || decay_bias.numel() != embed_dim) {
    throw std::invalid_argument("mamba2_scan_mixed decay_bias shape mismatch.");
  }
  if (!skip.defined() || skip.numel() != embed_dim) {
    throw std::invalid_argument("mamba2_scan_mixed skip shape mismatch.");
  }
  if (reset_mask.defined() &&
      (reset_mask.dim() != 2 || reset_mask.size(0) != projected.size(0) || reset_mask.size(1) != projected.size(1))) {
    throw std::invalid_argument("mamba2_scan_mixed reset_mask must have shape [batch, sequence].");
  }
}

void validate_step_inputs(
    const torch::Tensor& projected,
    const torch::Tensor& previous_scan,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip) {
  if (!projected.defined() || projected.dim() != 2) {
    throw std::invalid_argument("mamba2_step_mixed requires projected with shape [batch, 5 * dim].");
  }
  if (projected.size(1) % 5 != 0) {
    throw std::invalid_argument("mamba2_step_mixed projected last dimension must be divisible by 5.");
  }
  const auto embed_dim = projected.size(1) / 5;
  if (!previous_scan.defined() || previous_scan.sizes() != torch::IntArrayRef({projected.size(0), embed_dim})) {
    throw std::invalid_argument("mamba2_step_mixed previous_scan shape mismatch.");
  }
  if (!decay_bias.defined() || decay_bias.numel() != embed_dim) {
    throw std::invalid_argument("mamba2_step_mixed decay_bias shape mismatch.");
  }
  if (!skip.defined() || skip.numel() != embed_dim) {
    throw std::invalid_argument("mamba2_step_mixed skip shape mismatch.");
  }
}

}  // namespace

#ifdef PULSAR_HAS_MAMBA2_CUDA_KERNELS
std::tuple<torch::Tensor, torch::Tensor> mamba2_scan_forward_cuda(
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask);

std::vector<torch::Tensor> mamba2_scan_backward_cuda(
    const torch::Tensor& grad_mixed,
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask,
    const torch::Tensor& scan);

std::tuple<torch::Tensor, torch::Tensor> mamba2_step_forward_cuda(
    const torch::Tensor& projected,
    const torch::Tensor& previous_scan,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip);
#endif

#ifdef PULSAR_HAS_MAMBA2_CUDA_KERNELS
namespace {
class Mamba2ScanCudaFunction : public torch::autograd::Function<Mamba2ScanCudaFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor projected,
      torch::Tensor decay_bias,
      torch::Tensor skip,
      torch::Tensor reset_mask) {
    const bool has_reset = reset_mask.defined() && reset_mask.numel() > 0;
    projected = projected.contiguous();
    decay_bias = decay_bias.contiguous();
    skip = skip.contiguous();
    reset_mask = has_reset
        ? reset_mask.contiguous()
        : torch::empty({0}, projected.options());
    auto [mixed, scan] = mamba2_scan_forward_cuda(projected, decay_bias, skip, has_reset ? reset_mask : torch::Tensor{});
    ctx->saved_data["has_reset"] = has_reset;
    ctx->save_for_backward({projected, decay_bias, skip, reset_mask, scan});
    return mixed;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& projected = saved[0];
    const torch::Tensor& decay_bias = saved[1];
    const torch::Tensor& skip = saved[2];
    const bool has_reset = ctx->saved_data["has_reset"].toBool();
    const torch::Tensor reset_mask = has_reset ? saved[3] : torch::Tensor{};
    const torch::Tensor& scan = saved[4];
    std::vector<torch::Tensor> grads = mamba2_scan_backward_cuda(
        grad_outputs[0].contiguous(),
        projected,
        decay_bias,
        skip,
        reset_mask,
        scan);
    grads.push_back(torch::Tensor{});
    return grads;
  }
};
}  // namespace
#endif

#ifdef PULSAR_HAS_MAMBA2_HIP_KERNELS
std::tuple<torch::Tensor, torch::Tensor> mamba2_scan_forward_hip(
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask);

std::vector<torch::Tensor> mamba2_scan_backward_hip(
    const torch::Tensor& grad_mixed,
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask,
    const torch::Tensor& scan);

std::tuple<torch::Tensor, torch::Tensor> mamba2_step_forward_hip(
    const torch::Tensor& projected,
    const torch::Tensor& previous_scan,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip);
#endif

#ifdef PULSAR_HAS_MAMBA2_HIP_KERNELS
namespace {
class Mamba2ScanHipFunction : public torch::autograd::Function<Mamba2ScanHipFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor projected,
      torch::Tensor decay_bias,
      torch::Tensor skip,
      torch::Tensor reset_mask) {
    const bool has_reset = reset_mask.defined() && reset_mask.numel() > 0;
    projected = projected.contiguous();
    decay_bias = decay_bias.contiguous();
    skip = skip.contiguous();
    reset_mask = has_reset
        ? reset_mask.contiguous()
        : torch::empty({0}, projected.options());
    auto [mixed, scan] = mamba2_scan_forward_hip(projected, decay_bias, skip, has_reset ? reset_mask : torch::Tensor{});
    ctx->saved_data["has_reset"] = has_reset;
    ctx->save_for_backward({projected, decay_bias, skip, reset_mask, scan});
    return mixed;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& projected = saved[0];
    const torch::Tensor& decay_bias = saved[1];
    const torch::Tensor& skip = saved[2];
    const bool has_reset = ctx->saved_data["has_reset"].toBool();
    const torch::Tensor reset_mask = has_reset ? saved[3] : torch::Tensor{};
    const torch::Tensor& scan = saved[4];
    std::vector<torch::Tensor> grads = mamba2_scan_backward_hip(
        grad_outputs[0].contiguous(),
        projected,
        decay_bias,
        skip,
        reset_mask,
        scan);
    grads.push_back(torch::Tensor{});
    return grads;
  }
};
}  // namespace
#endif

namespace {

bool can_use_cuda_scan(const torch::Tensor& projected, const torch::Tensor& decay_bias, const torch::Tensor& skip) {
#if defined(PULSAR_HAS_MAMBA2_CUDA_KERNELS) || defined(PULSAR_HAS_MAMBA2_HIP_KERNELS)
  return projected.is_cuda() &&
      projected.scalar_type() == torch::kFloat32 &&
      decay_bias.scalar_type() == torch::kFloat32 &&
      skip.scalar_type() == torch::kFloat32;
#else
  (void)projected;
  (void)decay_bias;
  (void)skip;
  return false;
#endif
}

}  // namespace

bool mamba2_accelerator_kernels_available() {
#if defined(PULSAR_HAS_MAMBA2_CUDA_KERNELS) || defined(PULSAR_HAS_MAMBA2_HIP_KERNELS)
  return true;
#else
  return false;
#endif
}

bool mamba2_cuda_kernels_available() {
  return mamba2_accelerator_kernels_available();
}

torch::Tensor mamba2_scan_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask) {
  validate_scan_inputs(projected, decay_bias, skip, reset_mask);
#ifdef PULSAR_HAS_MAMBA2_CUDA_KERNELS
  if (can_use_cuda_scan(projected, decay_bias, skip)) {
    const torch::Tensor reset_arg = reset_mask.defined()
        ? reset_mask.to(projected.device()).to(torch::kFloat32)
        : torch::empty({0}, projected.options());
    return Mamba2ScanCudaFunction::apply(
        projected,
        decay_bias,
        skip,
        reset_arg);
  }
#endif
#ifdef PULSAR_HAS_MAMBA2_HIP_KERNELS
  if (can_use_cuda_scan(projected, decay_bias, skip)) {
    const torch::Tensor reset_arg = reset_mask.defined()
        ? reset_mask.to(projected.device()).to(torch::kFloat32)
        : torch::empty({0}, projected.options());
    return Mamba2ScanHipFunction::apply(
        projected,
        decay_bias,
        skip,
        reset_arg);
  }
#endif
  return fallback_mamba2_scan_mixed(projected, decay_bias, skip, reset_mask);
}

std::tuple<torch::Tensor, torch::Tensor> mamba2_step_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& previous_scan,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip) {
  validate_step_inputs(projected, previous_scan, decay_bias, skip);
#ifdef PULSAR_HAS_MAMBA2_CUDA_KERNELS
  if (!torch::autograd::GradMode::is_enabled() && can_use_cuda_scan(projected, decay_bias, skip)) {
    return mamba2_step_forward_cuda(
        projected.contiguous(),
        previous_scan.contiguous(),
        decay_bias.contiguous(),
        skip.contiguous());
  }
#endif
#ifdef PULSAR_HAS_MAMBA2_HIP_KERNELS
  if (!torch::autograd::GradMode::is_enabled() && can_use_cuda_scan(projected, decay_bias, skip)) {
    return mamba2_step_forward_hip(
        projected.contiguous(),
        previous_scan.contiguous(),
        decay_bias.contiguous(),
        skip.contiguous());
  }
#endif
  return fallback_mamba2_step_mixed(projected, previous_scan, decay_bias, skip);
}

}  // namespace pulsar

#endif  // PULSAR_HAS_TORCH
