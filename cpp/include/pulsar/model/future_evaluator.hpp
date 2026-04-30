#pragma once

#include <string>
#include <vector>

#include "pulsar/config/config.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>
#endif

namespace pulsar {

#ifdef PULSAR_HAS_TORCH

struct FutureEvaluationOutput {
  torch::Tensor embeddings;
  torch::Tensor outcome_logits;
  torch::Tensor delta_predictions;
};

class FutureEvaluatorImpl : public torch::nn::Module {
 public:
  explicit FutureEvaluatorImpl(FutureEvaluatorConfig config, int observation_dim);

  FutureEvaluationOutput forward_windows(torch::Tensor windows);
  torch::Tensor classify_embeddings(torch::Tensor embeddings);
  [[nodiscard]] const FutureEvaluatorConfig& config() const;
  [[nodiscard]] int observation_dim() const;
  [[nodiscard]] int max_horizon() const;

 private:
  FutureEvaluatorConfig config_{};
  int observation_dim_ = 0;
  int max_horizon_ = 0;
  torch::nn::Linear obs_proj_{nullptr};
  torch::nn::TransformerEncoder transformer_{nullptr};
  torch::nn::ModuleList embedding_heads_{};
  torch::nn::ModuleList outcome_heads_{};
  torch::nn::ModuleList delta_heads_{};
  torch::Tensor position_embedding_;
};

TORCH_MODULE(FutureEvaluator);

FutureEvaluator clone_future_evaluator(const FutureEvaluator& source, const torch::Device& device);
void ema_update_future_evaluator(const FutureEvaluator& target, const FutureEvaluator& source, float tau);

#else

struct FutureEvaluationOutput {};

class FutureEvaluator {
 public:
  FutureEvaluator() = default;
};

#endif

}  // namespace pulsar
