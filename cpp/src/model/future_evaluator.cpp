#include "pulsar/model/future_evaluator.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace pulsar {
namespace {

void validate_config(const FutureEvaluatorConfig& config, int observation_dim) {
  if (observation_dim <= 0) {
    throw std::invalid_argument("FutureEvaluator observation_dim must be positive.");
  }
  if (config.horizons.empty()) {
    throw std::invalid_argument("FutureEvaluatorConfig.horizons must not be empty.");
  }
  for (const int horizon : config.horizons) {
    if (horizon <= 0) {
      throw std::invalid_argument("FutureEvaluatorConfig.horizons must be positive.");
    }
  }
  if (config.latent_dim <= 0 || config.model_dim <= 0 || config.layers <= 0 ||
      config.heads <= 0 || config.feedforward_dim <= 0 || config.outcome_classes <= 0) {
    throw std::invalid_argument("FutureEvaluatorConfig dimensions must be positive.");
  }
  if (config.future_delta_loss_coef < 0.0F || !std::isfinite(config.future_delta_loss_coef)) {
    throw std::invalid_argument("FutureEvaluatorConfig.future_delta_loss_coef must be finite and non-negative.");
  }
}

}  // namespace

FutureEvaluatorImpl::FutureEvaluatorImpl(FutureEvaluatorConfig config, int observation_dim)
    : config_(std::move(config)), observation_dim_(observation_dim) {
  validate_config(config_, observation_dim_);
  max_horizon_ = *std::max_element(config_.horizons.begin(), config_.horizons.end());
  obs_proj_ = register_module("obs_proj", torch::nn::Linear(observation_dim_, config_.model_dim));

  auto layer_options = torch::nn::TransformerEncoderLayerOptions(config_.model_dim, config_.heads)
                           .dim_feedforward(config_.feedforward_dim)
                           .dropout(config_.dropout);
  transformer_ = register_module(
      "transformer",
      torch::nn::TransformerEncoder(torch::nn::TransformerEncoderLayer(layer_options), config_.layers));

  embedding_heads_ = register_module("embedding_heads", torch::nn::ModuleList());
  outcome_heads_ = register_module("outcome_heads", torch::nn::ModuleList());
  delta_heads_ = register_module("delta_heads", torch::nn::ModuleList());
  for (std::size_t i = 0; i < config_.horizons.size(); ++i) {
    embedding_heads_->push_back(torch::nn::Linear(config_.model_dim, config_.latent_dim));
    outcome_heads_->push_back(torch::nn::Linear(config_.latent_dim, config_.outcome_classes));
    delta_heads_->push_back(torch::nn::Linear(config_.latent_dim, observation_dim_));
  }

  position_embedding_ = register_parameter(
      "position_embedding",
      torch::randn({max_horizon_ + 1, config_.model_dim}) * 0.02);
}

FutureEvaluationOutput FutureEvaluatorImpl::forward_windows(torch::Tensor windows) {
  if (windows.dim() != 3 || windows.size(2) != observation_dim_) {
    throw std::invalid_argument("FutureEvaluator windows must have shape [batch, time, observation_dim].");
  }
  if (windows.size(1) < max_horizon_ + 1) {
    throw std::invalid_argument("FutureEvaluator window time dimension is shorter than max horizon.");
  }
  const auto batch = windows.size(0);

  std::vector<torch::Tensor> embeddings;
  std::vector<torch::Tensor> logits;
  std::vector<torch::Tensor> deltas;
  embeddings.reserve(config_.horizons.size());
  logits.reserve(config_.horizons.size());
  deltas.reserve(config_.horizons.size());
  for (std::size_t i = 0; i < config_.horizons.size(); ++i) {
    const int horizon = config_.horizons[i];
    const torch::Tensor projected = obs_proj_->forward(windows.narrow(1, 0, horizon + 1));
    const torch::Tensor sequence_first =
        (projected + position_embedding_.unsqueeze(0).narrow(1, 0, horizon + 1)).transpose(0, 1);
    const torch::Tensor encoded = transformer_->forward(sequence_first).transpose(0, 1);
    const torch::Tensor token = encoded.select(1, horizon);
    const torch::Tensor embedding =
        torch::tanh(embedding_heads_[i]->as<torch::nn::Linear>()->forward(token));
    embeddings.push_back(embedding);
    logits.push_back(outcome_heads_[i]->as<torch::nn::Linear>()->forward(embedding));
    deltas.push_back(delta_heads_[i]->as<torch::nn::Linear>()->forward(embedding));
  }

  return {
      torch::stack(embeddings, 1).reshape({batch, static_cast<std::int64_t>(config_.horizons.size()), config_.latent_dim}),
      torch::stack(logits, 1).reshape({batch, static_cast<std::int64_t>(config_.horizons.size()), config_.outcome_classes}),
      torch::stack(deltas, 1).reshape({batch, static_cast<std::int64_t>(config_.horizons.size()), observation_dim_}),
  };
}

torch::Tensor FutureEvaluatorImpl::classify_embeddings(torch::Tensor embeddings) {
  if (embeddings.size(-2) != static_cast<std::int64_t>(config_.horizons.size()) ||
      embeddings.size(-1) != config_.latent_dim) {
    throw std::invalid_argument("FutureEvaluator embeddings have incompatible shape.");
  }
  const auto leading = embeddings.sizes().vec();
  const torch::Tensor flat = embeddings.reshape({-1, static_cast<std::int64_t>(config_.horizons.size()), config_.latent_dim});
  std::vector<torch::Tensor> logits;
  logits.reserve(config_.horizons.size());
  for (std::size_t i = 0; i < config_.horizons.size(); ++i) {
    logits.push_back(outcome_heads_[i]->as<torch::nn::Linear>()->forward(flat.select(1, static_cast<long>(i))));
  }
  std::vector<std::int64_t> out_shape = leading;
  out_shape.back() = config_.outcome_classes;
  return torch::stack(logits, 1).reshape(out_shape);
}

const FutureEvaluatorConfig& FutureEvaluatorImpl::config() const {
  return config_;
}

int FutureEvaluatorImpl::observation_dim() const {
  return observation_dim_;
}

int FutureEvaluatorImpl::max_horizon() const {
  return max_horizon_;
}

FutureEvaluator clone_future_evaluator(const FutureEvaluator& source, const torch::Device& device) {
  if (!source) {
    return nullptr;
  }
  auto clone = FutureEvaluator(source->config(), source->observation_dim());
  torch::serialize::OutputArchive out;
  source->save(out);
  torch::serialize::InputArchive in;
  std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
  out.save_to(buffer);
  in.load_from(buffer, device);
  clone->load(in);
  clone->to(device);
  return clone;
}

void ema_update_future_evaluator(const FutureEvaluator& target, const FutureEvaluator& source, float tau) {
  if (!target || !source) {
    throw std::invalid_argument("ema_update_future_evaluator requires non-null evaluators.");
  }
  if (!std::isfinite(tau)) {
    throw std::invalid_argument("EMA tau must be finite.");
  }
  const float clamped_tau = std::clamp(tau, 0.0F, 1.0F);
  if (clamped_tau <= 0.0F) {
    return;
  }

  torch::NoGradGuard no_grad;
  const auto target_params = target->parameters();
  const auto source_params = source->parameters();
  if (target_params.size() != source_params.size()) {
    throw std::runtime_error("FutureEvaluator EMA parameter counts differ.");
  }
  for (std::size_t i = 0; i < target_params.size(); ++i) {
    target_params[i].mul_(1.0F - clamped_tau).add_(source_params[i], clamped_tau);
  }

  const auto target_buffers = target->buffers();
  const auto source_buffers = source->buffers();
  if (target_buffers.size() != source_buffers.size()) {
    throw std::runtime_error("FutureEvaluator EMA buffer counts differ.");
  }
  for (std::size_t i = 0; i < target_buffers.size(); ++i) {
    target_buffers[i].copy_(source_buffers[i]);
  }
}

}  // namespace pulsar

#endif
