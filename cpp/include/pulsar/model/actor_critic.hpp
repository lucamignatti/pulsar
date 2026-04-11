#pragma once

#include <memory>
#include <string>
#include <tuple>

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

struct PolicyOutput {
  torch::Tensor policy_logits;
  torch::Tensor value_logits;
  torch::Tensor expected_values;
  torch::Tensor sampled_values;
  torch::Tensor next_goal_logits;
  ContinuumState state;
};

struct SequenceOutput {
  torch::Tensor policy_logits;
  torch::Tensor value_logits;
  torch::Tensor expected_values;
  torch::Tensor sampled_values;
  torch::Tensor next_goal_logits;
  ContinuumState final_state;
};

class SharedActorCriticImpl : public torch::nn::Module {
 public:
  explicit SharedActorCriticImpl(ModelConfig config, PPOConfig ppo_config = {});

  [[nodiscard]] ContinuumState initial_state(std::int64_t batch_size, const torch::Device& device) const;
  PolicyOutput forward_step(torch::Tensor obs, ContinuumState state, torch::Tensor episode_starts = {});
  SequenceOutput forward_sequence(torch::Tensor obs_seq, ContinuumState state, torch::Tensor episode_starts = {});
  [[nodiscard]] torch::Tensor expected_value(torch::Tensor value_logits) const;
  [[nodiscard]] torch::Tensor sample_value(torch::Tensor value_logits) const;
  [[nodiscard]] torch::Tensor value_variance(torch::Tensor value_logits) const;
  [[nodiscard]] torch::Tensor value_entropy(torch::Tensor value_logits) const;
  [[nodiscard]] torch::Tensor support() const;
  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] const PPOConfig& ppo_config() const;

 private:
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

  ModelConfig config_{};
  PPOConfig ppo_config_{};
  torch::Tensor support_;
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
  torch::nn::Linear value_head_{nullptr};
  torch::nn::Linear next_goal_head_{nullptr};
  torch::Tensor ltm_basis_keys_;
  torch::Tensor ltm_basis_values_;
};

TORCH_MODULE(SharedActorCritic);

SharedActorCritic load_shared_model(const std::string& checkpoint_path, const std::string& device);

#else

struct ContinuumState {};
struct PolicyOutput {};
struct SequenceOutput {};

class SharedActorCritic {
 public:
  SharedActorCritic() = default;
};

#endif

}  // namespace pulsar
