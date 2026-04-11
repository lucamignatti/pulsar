#include "pulsar/model/next_goal_predictor.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

NextGoalPredictorImpl::NextGoalPredictorImpl(int input_dim, const NextGoalPredictorConfig& config)
    : input_dim_(input_dim), config_(config) {
  int hidden_in = input_dim_;
  for (const int hidden_size : config_.hidden_sizes) {
    backbone_->push_back(torch::nn::Linear(hidden_in, hidden_size));
    backbone_->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));
    backbone_->push_back(torch::nn::Functional(torch::relu));
    hidden_in = hidden_size;
  }

  output_head_ = register_module("output_head", torch::nn::Linear(hidden_in, config_.num_classes));
  register_module("backbone", backbone_);
}

torch::Tensor NextGoalPredictorImpl::forward(torch::Tensor obs) {
  return output_head_->forward(backbone_->forward(std::move(obs)));
}

int NextGoalPredictorImpl::input_dim() const {
  return input_dim_;
}

const NextGoalPredictorConfig& NextGoalPredictorImpl::config() const {
  return config_;
}

}  // namespace pulsar

#endif
