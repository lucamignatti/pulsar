#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/training/sparse_event_channels.hpp"

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
  torch::Tensor q_values;
};

struct ActorSequenceOutput {
  torch::Tensor policy_logits;
  torch::Tensor encoded;
  torch::Tensor value_win_logits;
  torch::Tensor features;
  torch::Tensor q_values;
};

/**
 * @brief LoRALinearImpl handles low-rank adaptations (A x B) of linear projections.
 * Used for evolutionary parameter search via EGGROLL (Sarkar et al., 2025).
 */
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

/**
 * @brief GoalCritic pair for contrastive self-supervised goal reaching (Nimonkar et al., 2025).
 * Represents state-action embeddings and goal embeddings trained via symmetric InfoNCE loss.
 */
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

/**
 * @brief Mamba2Block implements Structured State Space Duality (SSD) layer scan (Dao & Gu, 2024).
 */
class Mamba2BlockImpl : public torch::nn::Module {
 public:
  Mamba2BlockImpl(int embed_dim, int sequence_length, bool use_layer_norm);

  torch::Tensor forward(const torch::Tensor& tokens, const torch::Tensor& reset_mask = {});
  torch::Tensor forward_step(
      const torch::Tensor& token,
      const torch::Tensor& previous_conv_2,
      const torch::Tensor& previous_conv_1,
      const torch::Tensor& previous_scan,
      torch::Tensor* next_conv_2,
      torch::Tensor* next_conv_1,
      torch::Tensor* next_scan);

 public:
  int embed_dim_ = 0;
  int sequence_length_ = 0;
  bool use_layer_norm_ = true;
  torch::nn::LayerNorm input_norm_{nullptr};
  torch::nn::LayerNorm output_norm_{nullptr};
  torch::nn::Conv1d causal_conv_{nullptr};
  torch::nn::Linear input_projection_{nullptr};
  torch::nn::Linear output_projection_{nullptr};
  torch::Tensor decay_bias_;
  torch::Tensor skip_;
};

TORCH_MODULE(Mamba2Block);

/**
 * @brief Mamba2Encoder implements the sequence encoding backbone using stack of Mamba-2 blocks.
 */
class Mamba2EncoderImpl : public torch::nn::Module {
 public:
  explicit Mamba2EncoderImpl(const ModelConfig& config);

  torch::Tensor forward(const torch::Tensor& obs);
  torch::Tensor forward_sequence(const torch::Tensor& obs_seq, const torch::Tensor& episode_starts = {});
  torch::Tensor forward_step(
      const torch::Tensor& obs,
      const torch::Tensor& state,
      const torch::Tensor& episode_starts,
      torch::Tensor* next_state);
  [[nodiscard]] torch::Tensor initial_state(int64_t batch, const torch::Device& device) const;

 private:
  int observation_dim_ = 0;
  int embed_dim_ = 0;
  int sequence_length_ = 1;
  torch::nn::Linear input_projection_{nullptr};
  std::vector<Mamba2Block> blocks_{};
  torch::nn::LayerNorm output_norm_{nullptr};
};

TORCH_MODULE(Mamba2Encoder);

/**
 * @brief SparseRewardPredictor live online auxiliary MLP.
 */
class SparseRewardPredictorImpl : public torch::nn::Module {
 public:
  explicit SparseRewardPredictorImpl(int feature_dim) {
    net_ = torch::nn::Sequential(
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({feature_dim})),
        torch::nn::Linear(feature_dim, 128),
        torch::nn::ReLU(),
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})),
        torch::nn::Linear(128, 2)
    );
    register_module("net", net_);
    auto& output = net_->at<torch::nn::LinearImpl>(4);
    torch::nn::init::normal_(output.weight, 0.0, 0.01);
    torch::nn::init::constant_(output.bias, 0.0);
  }

  torch::Tensor forward(torch::Tensor x) const {
    auto* non_const_net = const_cast<torch::nn::SequentialImpl*>(net_.get());
    return torch::clamp(non_const_net->forward(x), -10.0, 10.0);
  }

  const torch::nn::LayerNormImpl& input_norm() const {
    auto* non_const_net = const_cast<torch::nn::SequentialImpl*>(net_.get());
    return non_const_net->at<torch::nn::LayerNormImpl>(0);
  }

  const torch::nn::LinearImpl& input_linear() const {
    auto* non_const_net = const_cast<torch::nn::SequentialImpl*>(net_.get());
    return non_const_net->at<torch::nn::LinearImpl>(1);
  }

  const torch::nn::LayerNormImpl& hidden_norm() const {
    auto* non_const_net = const_cast<torch::nn::SequentialImpl*>(net_.get());
    return non_const_net->at<torch::nn::LayerNormImpl>(3);
  }

  const torch::nn::LinearImpl& output_linear() const {
    auto* non_const_net = const_cast<torch::nn::SequentialImpl*>(net_.get());
    return non_const_net->at<torch::nn::LinearImpl>(4);
  }

 private:
  torch::nn::Sequential net_{nullptr};
};

TORCH_MODULE(SparseRewardPredictor);

/**
 * @brief VRPOActorImpl is a unified network for Variance-Reduced Policy Optimization.
 * Houses the SSD sequence encoder, the EGGROLL-perturbed LoRA policy head, and centralized Q-head.
 */
class VRPOActorImpl : public torch::nn::Module {
 public:
  explicit VRPOActorImpl(
      ModelConfig config,
      const GoalCriticConfig& goal_critic_config = {},
      const ESLoraConfig& es_lora_config = {},
      const VRPOConfig& vrpo_config = {});

  ActorStepOutput forward_step(
      torch::Tensor obs,
      torch::Tensor goal_values = {},
      bool compute_value = true);
  ActorStepOutput forward_step_stateful(
      torch::Tensor obs,
      torch::Tensor state,
      torch::Tensor episode_starts,
      torch::Tensor* next_state,
      torch::Tensor goal_values = {},
      bool compute_value = true);
  ActorSequenceOutput forward_sequence(
      torch::Tensor obs_seq,
      torch::Tensor goal_values = {},
      torch::Tensor episode_starts = {},
      bool compute_value = true);
  [[nodiscard]] torch::Tensor initial_recurrent_state(int64_t batch, const torch::Device& device) const;
  [[nodiscard]] int feature_dim() const;
  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] const GoalCriticConfig& goal_critic_config() const;
  [[nodiscard]] const ESLoraConfig& es_lora_config() const;
  [[nodiscard]] const VRPOConfig& vrpo_config() const;
  [[nodiscard]] std::vector<std::string> enabled_critic_heads() const;
  torch::Tensor value_head_forward(const torch::Tensor& encoded);
  torch::Tensor q_head_forward(const torch::Tensor& encoded);
  torch::Tensor policy_head_forward(const torch::Tensor& features);
  SparseRewardPredictor& predictor_head(std::size_t channel_index);
  const SparseRewardPredictor& predictor_head(std::size_t channel_index) const;
  torch::Tensor forward_all_predictors(const torch::Tensor& features) const;

  // EGGROLL Evolutionary LoRA Perturbations
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

  ModelConfig config_{};
  GoalCriticConfig goal_critic_config_{};
  ESLoraConfig es_lora_config_{};
  VRPOConfig vrpo_config_{};
  int feature_dim_ = 0;
  Mamba2Encoder mamba2_encoder_{nullptr};

  torch::nn::Sequential policy_hidden_{nullptr};
  LoRALinear policy_lora_{nullptr};

  torch::nn::Sequential q_head_{nullptr};
  GoalCritic goal_critic_{nullptr};

  std::vector<SparseRewardPredictor> predictor_heads_{};
  std::vector<std::uint8_t> predictor_active_{};
};

TORCH_MODULE(VRPOActor);

VRPOActor load_vrpo_actor(const std::string& checkpoint_path, const std::string& device);
VRPOActor clone_vrpo_actor(const VRPOActor& source, const torch::Device& device);
void copy_vrpo_actor_tensors_to(const VRPOActor& source, VRPOActor& target, const torch::Device& device);

#else

struct ActorStepOutput {};
struct ActorSequenceOutput {};

class VRPOActor {
 public:
  VRPOActor() = default;
};

#endif

}  // namespace pulsar
