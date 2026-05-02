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

struct ActorStepOutput {
  torch::Tensor policy_logits;
  torch::Tensor value_logits;
  torch::Tensor features;
  ContinuumState state;
};

struct ActorSequenceOutput {
  torch::Tensor policy_logits;
  torch::Tensor value_logits;
  torch::Tensor features;
  ContinuumState final_state;
};

class PPOActorImpl : public torch::nn::Module {
 public:
  explicit PPOActorImpl(ModelConfig config);

  [[nodiscard]] ContinuumState initial_state(std::int64_t batch_size, const torch::Device& device) const;
  ActorStepOutput forward_step(torch::Tensor obs, ContinuumState state, torch::Tensor episode_starts = {});
  ActorSequenceOutput forward_sequence(torch::Tensor obs_seq, ContinuumState state, torch::Tensor episode_starts = {});
  [[nodiscard]] torch::Tensor value_support() const;
  [[nodiscard]] int feature_dim() const;
  [[nodiscard]] const ModelConfig& config() const;

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
  torch::nn::Linear value_head_{nullptr};
  torch::Tensor ltm_basis_keys_;
  torch::Tensor ltm_basis_values_;
  torch::Tensor atom_support_;
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
