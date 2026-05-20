#include <utility>

#include <torch/torch.h>

namespace pulsar {

#ifdef PULSAR_HAS_LAYERNORM_CUDA_KERNELS
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_layer_norm_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps);

torch::Tensor fused_layer_norm_inference_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps);

std::vector<torch::Tensor> fused_layer_norm_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& saved_mean,
    const torch::Tensor& saved_inv_std);
#endif

#ifdef PULSAR_HAS_LAYERNORM_HIP_KERNELS
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_layer_norm_forward_hip(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps);

torch::Tensor fused_layer_norm_inference_hip(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps);

std::vector<torch::Tensor> fused_layer_norm_backward_hip(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& saved_mean,
    const torch::Tensor& saved_inv_std);
#endif
namespace {

#if defined(PULSAR_HAS_LAYERNORM_CUDA_KERNELS)
class FusedLayerNormCudaFunction : public torch::autograd::Function<FusedLayerNormCudaFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor bias,
      float eps) {
    input = input.contiguous().to(torch::kFloat32);
    weight = weight.defined() ? weight.contiguous().to(torch::kFloat32) : weight;
    bias = bias.defined() ? bias.contiguous().to(torch::kFloat32) : bias;
    auto [output, mean, inv_std] = fused_layer_norm_forward_cuda(input, weight, bias, eps);
    ctx->saved_data["has_weight"] = weight.defined();
    ctx->saved_data["has_bias"] = bias.defined();
    ctx->save_for_backward({input, weight.defined() ? weight : torch::Tensor{}, bias.defined() ? bias : torch::Tensor{}, mean, inv_std});
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& input = saved[0];
    const torch::Tensor weight = ctx->saved_data["has_weight"].toBool() ? saved[1] : torch::Tensor{};
    const torch::Tensor bias = ctx->saved_data["has_bias"].toBool() ? saved[2] : torch::Tensor{};
    const torch::Tensor& mean = saved[3];
    const torch::Tensor& inv_std = saved[4];
    auto grads = fused_layer_norm_backward_cuda(
        grad_outputs[0].contiguous().to(torch::kFloat32),
        input,
        weight,
        bias,
        mean,
        inv_std);
    grads.push_back(torch::Tensor{});  // grad for eps
    return grads;
  }
};
#elif defined(PULSAR_HAS_LAYERNORM_HIP_KERNELS)
class FusedLayerNormHipFunction : public torch::autograd::Function<FusedLayerNormHipFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor bias,
      float eps) {
    input = input.contiguous().to(torch::kFloat32);
    weight = weight.defined() ? weight.contiguous().to(torch::kFloat32) : weight;
    bias = bias.defined() ? bias.contiguous().to(torch::kFloat32) : bias;
    auto [output, mean, inv_std] = fused_layer_norm_forward_hip(input, weight, bias, eps);
    ctx->saved_data["has_weight"] = weight.defined();
    ctx->saved_data["has_bias"] = bias.defined();
    ctx->save_for_backward({input, weight.defined() ? weight : torch::Tensor{}, bias.defined() ? bias : torch::Tensor{}, mean, inv_std});
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& input = saved[0];
    const torch::Tensor weight = ctx->saved_data["has_weight"].toBool() ? saved[1] : torch::Tensor{};
    const torch::Tensor bias = ctx->saved_data["has_bias"].toBool() ? saved[2] : torch::Tensor{};
    const torch::Tensor& mean = saved[3];
    const torch::Tensor& inv_std = saved[4];
    auto grads = fused_layer_norm_backward_hip(
        grad_outputs[0].contiguous().to(torch::kFloat32),
        input,
        weight,
        bias,
        mean,
        inv_std);
    grads.push_back(torch::Tensor{});  // grad for eps
    return grads;
  }
};
#endif

constexpr float kLayerNormEps = 1.0e-5F;
constexpr int64_t kLayerNormAccelMinElements = 65536;

bool should_use_fused_layer_norm_accel(const torch::Tensor& input) {
  return input.is_cuda() &&
      input.scalar_type() == torch::kFloat32 &&
      input.numel() >= kLayerNormAccelMinElements &&
      input.size(-1) >= 64;
}

torch::Tensor stable_layer_norm_fallback(const torch::Tensor& input, const torch::nn::LayerNorm& norm) {
  const torch::Tensor input_f32 = input.scalar_type() == torch::kFloat32 ? input : input.to(torch::kFloat32);
  const int64_t normalized_dims = norm->weight.defined() ? norm->weight.dim() : 1;
  std::vector<int64_t> reduce_dims;
  reduce_dims.reserve(static_cast<std::size_t>(normalized_dims));
  for (int64_t dim = input_f32.dim() - normalized_dims; dim < input_f32.dim(); ++dim) {
    reduce_dims.push_back(dim);
  }

  const torch::Tensor mean = input_f32.mean(reduce_dims, /*keepdim=*/true);
  const torch::Tensor centered = input_f32 - mean;
  const torch::Tensor inv_std = (centered.square().mean(reduce_dims, /*keepdim=*/true) + kLayerNormEps).rsqrt();
  torch::Tensor output = centered * inv_std;

  if (norm->weight.defined()) {
    std::vector<int64_t> affine_shape(static_cast<std::size_t>(input_f32.dim()), 1);
    const auto weight_sizes = norm->weight.sizes();
    for (int64_t dim = 0; dim < normalized_dims; ++dim) {
      affine_shape[static_cast<std::size_t>(input_f32.dim() - normalized_dims + dim)] = weight_sizes[dim];
    }
    output = output * norm->weight.view(affine_shape);
    if (norm->bias.defined()) {
      output = output + norm->bias.view(affine_shape);
    }
  }

  return input.scalar_type() == torch::kFloat32 ? output : output.to(input.scalar_type());
}

}  // namespace

torch::Tensor fused_layer_norm(const torch::Tensor& input, const torch::nn::LayerNorm& norm) {
#if defined(PULSAR_HAS_LAYERNORM_CUDA_KERNELS)
  if (should_use_fused_layer_norm_accel(input)) {
    torch::Tensor input_f32 = input.contiguous().to(torch::kFloat32);
    torch::Tensor weight = norm->weight.defined()
        ? norm->weight
        : torch::Tensor{};
    torch::Tensor bias = norm->bias.defined()
        ? norm->bias
        : torch::Tensor{};
    weight = weight.defined() ? weight.contiguous().to(torch::kFloat32) : weight;
    bias = bias.defined() ? bias.contiguous().to(torch::kFloat32) : bias;
    const bool needs_grad =
        input.requires_grad() ||
        (weight.defined() && weight.requires_grad()) ||
        (bias.defined() && bias.requires_grad());
    if (!torch::GradMode::is_enabled() || !needs_grad) {
      torch::Tensor output = fused_layer_norm_inference_cuda(input_f32, weight, bias, kLayerNormEps);
      return input.scalar_type() == torch::kFloat32 ? output : output.to(input.scalar_type());
    }
    return FusedLayerNormCudaFunction::apply(input_f32, weight, bias, kLayerNormEps);
  }
#elif defined(PULSAR_HAS_LAYERNORM_HIP_KERNELS)
  if (should_use_fused_layer_norm_accel(input)) {
    torch::Tensor input_f32 = input.contiguous().to(torch::kFloat32);
    torch::Tensor weight = norm->weight.defined()
        ? norm->weight
        : torch::Tensor{};
    torch::Tensor bias = norm->bias.defined()
        ? norm->bias
        : torch::Tensor{};
    weight = weight.defined() ? weight.contiguous().to(torch::kFloat32) : weight;
    bias = bias.defined() ? bias.contiguous().to(torch::kFloat32) : bias;
    const bool needs_grad =
        input.requires_grad() ||
        (weight.defined() && weight.requires_grad()) ||
        (bias.defined() && bias.requires_grad());
    if (!torch::GradMode::is_enabled() || !needs_grad) {
      torch::Tensor output = fused_layer_norm_inference_hip(input_f32, weight, bias, kLayerNormEps);
      return input.scalar_type() == torch::kFloat32 ? output : output.to(input.scalar_type());
    }
    return FusedLayerNormHipFunction::apply(input_f32, weight, bias, kLayerNormEps);
  }
#endif
  return stable_layer_norm_fallback(input, norm);
}

}  // namespace pulsar
