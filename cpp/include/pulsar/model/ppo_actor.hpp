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

struct ContinuumState {
  torch::Tensor workspace;
  torch::Tensor stm_keys;
  torch::Tensor stm_values;
  torch::Tensor stm_strengths;
  torch::Tensor stm_write_index;
  torch::Tensor ltm_coeffs;
  torch::Tensor timestep;
};

struct ActorStepOutput {
  torch::Tensor policy_logits;
  torch::Tensor encoded;
  torch::Tensor value_win_logits;
  torch::Tensor goal_critic_logits;
  torch::Tensor features;
  ContinuumState state;
};

struct ActorSequenceOutput {
  torch::Tensor policy_logits;
  torch::Tensor encoded;
  torch::Tensor value_win_logits;
  torch::Tensor goal_critic_logits;
  torch::Tensor features;
  ContinuumState final_state;
};

class LoRALinearImpl : public torch::nn::Module {
 public:
  LoRALinearImpl(int in_features, int out_features, int rank, float lora_alpha = 4.0F);

  torch::Tensor forward(torch::Tensor x);
  void reset_lora_parameters();

  torch::nn::Linear base{nullptr};
  torch::Tensor A;
  torch::Tensor B;

  [[nodiscard]] std::vector<torch::Tensor> lora_parameters() const;
  [[nodiscard]] std::vector<torch::Tensor> lora_parameters_flat() const;
  void restore_lora_parameters(const std::vector<torch::Tensor>& params);
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
  GoalCriticImpl(int feature_dim, int action_dim, int num_atoms, int hidden_dim = 256);

  torch::Tensor forward(
      const torch::Tensor& features,
      const torch::Tensor& action_ids,
      const torch::Tensor& goal_value);

 private:
  torch::nn::Linear input_proj_{nullptr};
  torch::nn::Linear output_proj_{nullptr};
  int action_dim_;
  int hidden_dim_;
};

TORCH_MODULE(GoalCritic);

class PPOActorImpl : public torch::nn::Module {
 public:
  explicit PPOActorImpl(ModelConfig config, const GoalCriticConfig& goal_critic_config = {});

  [[nodiscard]] ContinuumState initial_state(std::int64_t batch_size, const torch::Device& device) const;
  ActorStepOutput forward_step(torch::Tensor obs, ContinuumState state, torch::Tensor episode_starts = {});
  ActorSequenceOutput forward_sequence(torch::Tensor obs_seq, ContinuumState state, torch::Tensor episode_starts = {});
  [[nodiscard]] torch::Tensor value_win_support() const;
  [[nodiscard]] torch::Tensor goal_critic_support() const;
  [[nodiscard]] int feature_dim() const;
  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] const GoalCriticConfig& goal_critic_config() const;
  [[nodiscard]] std::vector<std::string> enabled_critic_heads() const;

  [[nodiscard]] std::vector<torch::Tensor> es_lora_parameters() const;
  [[nodiscard]] std::vector<torch::Tensor> es_lora_parameters_flat() const;
  void restore_es_lora_parameters(const std::vector<torch::Tensor>& params);
  void apply_lora_perturbation(const std::vector<torch::Tensor>& perturbation, float sigma);
  [[nodiscard]] const LoRALinear& policy_lora() const;
  [[nodiscard]] GoalCritic& goal_critic();

 private:
  ActorStepOutput forward_encoded_step(torch::Tensor encoded, ContinuumState state, torch::Tensor episode_starts = {});
  [[nodiscard]] ContinuumState apply_episode_starts(ContinuumState state, torch::Tensor episode_starts) const;
  [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> read_memories(
      const torch::Tensor& encoded,
      const ContinuumState& state);
  [[nodiscard]] ContinuumState write_short_term_memory(
      ContinuumState state,
      const torch::Tensor& key,
      const torch::Tensor& value);
  [[nodiscard]] ContinuumState maybe_consolidate(
      ContinuumState state,
      const torch::Tensor& controller_hidden);
  [[nodiscard]] torch::nn::Sequential make_value_win_head(int input_dim) const;

  ModelConfig config_{};
  GoalCriticConfig goal_critic_config_{};
  int feature_dim_ = 0;
  torch::nn::Sequential encoder_{};
  torch::nn::Linear query_proj_{nullptr};
  torch::nn::Linear stm_context_proj_{nullptr};
  torch::nn::Linear ltm_query_proj_{nullptr};
  torch::nn::Linear ltm_context_proj_{nullptr};
  torch::nn::Linear ltm_write_proj_{nullptr};
  torch::nn::Linear ltm_gate_proj_{nullptr};
  torch::nn::Linear gate_proj_{nullptr};
  torch::nn::Linear controller_proj_{nullptr};
  torch::nn::GRUCell workspace_cell_{nullptr};
  torch::nn::Linear stm_key_write_{nullptr};
  torch::nn::Linear stm_value_write_{nullptr};

  torch::nn::Sequential policy_hidden_{nullptr};
  LoRALinear policy_lora_{nullptr};

  torch::nn::Sequential value_head_win_{nullptr};
  torch::Tensor atom_support_win_;
  GoalCritic goal_critic_{nullptr};
  torch::Tensor atom_support_goal_;

  torch::Tensor ltm_basis_keys_;
  torch::Tensor ltm_basis_values_;
};

TORCH_MODULE(PPOActor);

PPOActor load_ppo_actor(const std::string& checkpoint_path, const std::string& device);
PPOActor clone_ppo_actor(const PPOActor& source, const torch::Device& device);

#else

struct ContinuumState {};
struct ActorStepOutput {};
struct ActorSequenceOutput {};

class PPOActor {
 public:
  PPOActor() = default;
};

#endif

}  // namespace pulsar
