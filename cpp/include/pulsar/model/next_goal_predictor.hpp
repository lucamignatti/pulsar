#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pulsar/config/config.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>
#endif

namespace pulsar {

#ifdef PULSAR_HAS_TORCH

class NextGoalPredictorImpl : public torch::nn::Module {
 public:
  NextGoalPredictorImpl(int input_dim, const NextGoalPredictorConfig& config);

  torch::Tensor forward(torch::Tensor obs);
  [[nodiscard]] int input_dim() const;
  [[nodiscard]] const NextGoalPredictorConfig& config() const;

 private:
  int input_dim_ = 0;
  NextGoalPredictorConfig config_{};
  torch::nn::Sequential backbone_{};
  torch::nn::Linear output_head_{nullptr};
};

TORCH_MODULE(NextGoalPredictor);

#else

class NextGoalPredictor {
 public:
  NextGoalPredictor() = default;
};

#endif

}  // namespace pulsar
