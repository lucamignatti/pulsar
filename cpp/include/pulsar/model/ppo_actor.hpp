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

struct ValueHeadOutput {
  torch::Tensor logits;
  torch::Tensor support;
};

struct ActorStepOutput {
  torch::Tensor policy_logits;
  torch::Tensor encoded;
  ValueHeadOutput value_ext;
  ValueHeadOutput value_cur;
  ValueHeadOutput value_learn;
  ValueHeadOutput value_ctrl;
  torch::Tensor features;
  ContinuumState state;
};

struct ActorSequenceOutput {
  torch::Tensor policy_logits;
  torch::Tensor encoded;
  ValueHeadOutput value_ext;
  ValueHeadOutput value_cur;
  ValueHeadOutput value_learn;
  ValueHeadOutput value_ctrl;
  torch::Tensor features;
  ContinuumState final_state;
};

class PPOActorImpl : public torch::nn::Module {
 public:
  explicit PPOActorImpl(ModelConfig config);

  [[nodiscard]] ContinuumState initial_state(std::int64_t batch_size, const torch::Device& device) const;
  ActorStepOutput forward_step(torch::Tensor obs, ContinuumState state, torch::Tensor episode_starts = {});
  ActorSequenceOutput forward_sequence(torch::Tensor obs_seq, ContinuumState state, torch::Tensor episode_starts = {});
  [[nodiscard]] torch::Tensor value_support(const std::string& head_name = "extrinsic") const;
  [[nodiscard]] const ValueHeadOutput& value_head_output(const std::string& head_name, const ActorStepOutput& output) const;
  [[nodiscard]] const ValueHeadOutput& value_head_output(const std::string& head_name, const ActorSequenceOutput& output) const;
  [[nodiscard]] std::vector<std::string> enabled_critic_heads() const;
  [[nodiscard]] int feature_dim() const;
  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] torch::Tensor forward_predict_next(torch::Tensor encoded, torch::Tensor actions);
  [[nodiscard]] torch::Tensor forward_predict_action(torch::Tensor encoded_t, torch::Tensor encoded_tp1);
  [[nodiscard]] torch::Tensor compute_forward_prediction_error(torch::Tensor encoded, torch::Tensor actions, torch::Tensor encoded_tp1);

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
  [[nodiscard]] torch::nn::Sequential make_value_head(int input_dim, const CriticHeadConfig& head_config) const;
  [[nodiscard]] torch::nn::Sequential make_forward_head(int input_dim, int forward_action_dim) const;
  [[nodiscard]] torch::nn::Sequential make_inverse_head() const;
  [[nodiscard]] torch::Tensor make_atom_support(float v_min, float v_max, int num_atoms) const;
  void build_value_head(const std::string& name, torch::nn::Sequential& head, torch::Tensor& support, const CriticHeadConfig& head_cfg);

  ModelConfig config_{};
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
  torch::nn::Linear policy_head_{nullptr};

  torch::nn::Sequential value_head_ext_{nullptr};
  torch::nn::Sequential value_head_cur_{nullptr};
  torch::nn::Sequential value_head_learn_{nullptr};
  torch::nn::Sequential value_head_ctrl_{nullptr};

  torch::nn::Sequential forward_head_{nullptr};
  torch::nn::Sequential inverse_head_{nullptr};

  torch::Tensor atom_support_ext_;
  torch::Tensor atom_support_cur_;
  torch::Tensor atom_support_learn_;
  torch::Tensor atom_support_ctrl_;

  torch::Tensor ltm_basis_keys_;
  torch::Tensor ltm_basis_values_;

  std::unordered_map<std::string, torch::nn::Sequential*> value_heads_;
  std::unordered_map<std::string, torch::Tensor*> atom_supports_;
  std::vector<std::string> enabled_critic_heads_;
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
