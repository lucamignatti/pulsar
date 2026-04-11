#include "pulsar/model/actor_critic.hpp"

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <stdexcept>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"

namespace pulsar {

SharedActorCriticImpl::SharedActorCriticImpl(ModelConfig config) : config_(std::move(config)) {
  int in_features = config_.observation_dim;

  for (const int hidden_size : config_.hidden_sizes) {
    backbone_->push_back(torch::nn::Linear(in_features, hidden_size));
    if (config_.use_layer_norm) {
      backbone_->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));
    }
    backbone_->push_back(torch::nn::Functional(torch::relu));
    in_features = hidden_size;
  }

  policy_head_ = register_module("policy_head", torch::nn::Linear(in_features, config_.action_dim));
  value_head_ = register_module("value_head", torch::nn::Linear(in_features, 1));
  register_module("backbone", backbone_);
}

PolicyOutput SharedActorCriticImpl::forward(torch::Tensor obs) {
  torch::Tensor hidden = backbone_->forward(std::move(obs));
  return {
      policy_head_->forward(hidden),
      value_head_->forward(hidden).squeeze(-1),
  };
}

const ModelConfig& SharedActorCriticImpl::config() const {
  return config_;
}

SharedActorCritic load_shared_model(const std::string& checkpoint_path, const std::string& device) {
  namespace fs = std::filesystem;

  const fs::path base(checkpoint_path);
  const ExperimentConfig config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_checkpoint_metadata(metadata, config);

  SharedActorCritic model(config.model);
  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string());
  model->load(archive);
  model->to(torch::Device(device));
  model->eval();
  return model;
}

}  // namespace pulsar

#endif
