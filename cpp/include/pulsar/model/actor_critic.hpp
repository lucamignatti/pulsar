#pragma once

#include <memory>
#include <string>

#include "pulsar/config/config.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>
#endif

namespace pulsar {

#ifdef PULSAR_HAS_TORCH

struct PolicyOutput {
  torch::Tensor logits;
  torch::Tensor values;
};

class SharedActorCriticImpl : public torch::nn::Module {
 public:
  explicit SharedActorCriticImpl(ModelConfig config);

  PolicyOutput forward(torch::Tensor obs);
  [[nodiscard]] const ModelConfig& config() const;

 private:
  ModelConfig config_{};
  torch::nn::Sequential backbone_{};
  torch::nn::Linear policy_head_{nullptr};
  torch::nn::Linear value_head_{nullptr};
};

TORCH_MODULE(SharedActorCritic);

SharedActorCritic load_shared_model(const std::string& checkpoint_path, const std::string& device);

#else

struct PolicyOutput {};

class SharedActorCritic {
 public:
  SharedActorCritic() = default;
};

#endif

}  // namespace pulsar
