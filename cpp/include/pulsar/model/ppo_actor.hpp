#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "pulsar/config/config.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>
#endif

namespace pulsar {

#ifdef PULSAR_HAS_TORCH

struct ActorStepOutput {
  torch::Tensor policy_logits;
  torch::Tensor encoded;
  torch::Tensor value_win_logits;
  torch::Tensor features;
};

struct ActorSequenceOutput {
  torch::Tensor policy_logits;
  torch::Tensor encoded;
  torch::Tensor value_win_logits;
  torch::Tensor features;
};

class LoRALinearImpl : public torch::nn::Module {
 public:
  LoRALinearImpl(int in_features, int out_features, int rank, float lora_alpha = 4.0F);

  torch::Tensor forward(torch::Tensor x);
  torch::Tensor forward_eggroll_population(
      torch::Tensor x,
      const torch::Tensor& A_stack,
      const torch::Tensor& B_stack,
      float sigma);
  void reset_lora_parameters();

  torch::nn::Linear base{nullptr};
  torch::Tensor A;
  torch::Tensor B;

  [[nodiscard]] std::vector<torch::Tensor> lora_parameters() const;
  [[nodiscard]] std::vector<torch::Tensor> lora_parameters_flat() const;
  void restore_lora_parameters(const std::vector<torch::Tensor>& params);
  void apply_base_weight_update(const torch::Tensor& delta_weight);
  [[nodiscard]] int in_features() const;
  [[nodiscard]] int out_features() const;
  [[nodiscard]] int rank() const;
  [[nodiscard]] float scale() const;

 private:
  int rank_;
  float scale_;
};

TORCH_MODULE(LoRALinear);

class GoalCriticImpl : public torch::nn::Module {
 public:
  GoalCriticImpl(int feature_dim, int action_dim, int embedding_dim = 64, int hidden_dim = 256, int goal_dim = 3);

  torch::Tensor sa_embedding(const torch::Tensor& features, const torch::Tensor& action_inputs);
  torch::Tensor goal_embedding(const torch::Tensor& goal_values);
  torch::Tensor forward(
      const torch::Tensor& features,
      const torch::Tensor& action_inputs,
      const torch::Tensor& goal_values);

  int goal_dim() const { return goal_dim_; }

 private:
  torch::nn::Sequential sa_encoder_{nullptr};
  torch::nn::Sequential goal_encoder_{nullptr};
  int action_dim_;
  int hidden_dim_;
  int embedding_dim_;
  int goal_dim_;
};

TORCH_MODULE(GoalCritic);

class PPOActorImpl : public torch::nn::Module {
 public:
  explicit PPOActorImpl(
      ModelConfig config,
      const GoalCriticConfig& goal_critic_config = {},
      const ESLoraConfig& es_lora_config = {});

  ActorStepOutput forward_step(
      torch::Tensor obs,
      torch::Tensor goal_values = {});
  ActorSequenceOutput forward_sequence(
      torch::Tensor obs_seq,
      torch::Tensor goal_values = {});
  [[nodiscard]] int feature_dim() const;
  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] const GoalCriticConfig& goal_critic_config() const;
  [[nodiscard]] const ESLoraConfig& es_lora_config() const;
  [[nodiscard]] std::vector<std::string> enabled_critic_heads() const;

  [[nodiscard]] std::vector<torch::Tensor> es_lora_parameters() const;
  [[nodiscard]] std::vector<torch::Tensor> es_lora_parameters_flat() const;
  void restore_es_lora_parameters(const std::vector<torch::Tensor>& params);
  void apply_lora_perturbation(const std::vector<torch::Tensor>& perturbation, float sigma);
  [[nodiscard]] torch::Tensor policy_eggroll_logits(
      const torch::Tensor& features,
      const torch::Tensor& A_stack,
      const torch::Tensor& B_stack,
      float sigma,
      torch::Tensor goal_values = {});
  void apply_policy_eggroll_update(const torch::Tensor& delta_weight);
  [[nodiscard]] const LoRALinear& policy_lora() const;
  [[nodiscard]] GoalCritic& goal_critic();

 private:
  [[nodiscard]] torch::nn::Sequential make_value_win_head(int input_dim) const;

  ModelConfig config_{};
  GoalCriticConfig goal_critic_config_{};
  ESLoraConfig es_lora_config_{};
  int feature_dim_ = 0;
  torch::nn::Sequential encoder_{};

  torch::nn::Sequential policy_hidden_{nullptr};
  LoRALinear policy_lora_{nullptr};

  torch::nn::Sequential value_head_win_{nullptr};
  GoalCritic goal_critic_{nullptr};
};

TORCH_MODULE(PPOActor);

PPOActor load_ppo_actor(const std::string& checkpoint_path, const std::string& device);
PPOActor clone_ppo_actor(const PPOActor& source, const torch::Device& device);

#else

struct ActorStepOutput {};
struct ActorSequenceOutput {};

class PPOActor {
 public:
  PPOActor() = default;
};

#endif

}  // namespace pulsar
