#include "pulsar/model/actor_critic.hpp"

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <stdexcept>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"

namespace pulsar {
namespace {

torch::Tensor sample_categorical_from_logits(const torch::Tensor& logits) {
  const torch::Tensor uniform = torch::rand_like(logits).clamp_(1.0e-6, 1.0 - 1.0e-6);
  const torch::Tensor gumbel = -torch::log(-torch::log(uniform));
  return (logits + gumbel).argmax(-1);
}

torch::Tensor maybe_zero_mask(torch::Tensor tensor, const torch::Tensor& mask) {
  if (!mask.defined()) {
    return tensor;
  }
  torch::Tensor expanded_mask = mask;
  while (expanded_mask.dim() < tensor.dim()) {
    expanded_mask = expanded_mask.unsqueeze(-1);
  }
  return tensor * (1.0 - expanded_mask);
}

torch::Tensor masked_softmax(torch::Tensor logits, const torch::Tensor& strengths) {
  const torch::Tensor masked_logits = logits.masked_fill(strengths <= 1.0e-6, -1.0e9);
  return torch::softmax(masked_logits, -1);
}

}  // namespace

SharedActorCriticImpl::SharedActorCriticImpl(ModelConfig config, PPOConfig ppo_config)
    : config_(std::move(config)), ppo_config_(std::move(ppo_config)) {
  encoder_->push_back(torch::nn::Linear(config_.observation_dim, config_.encoder_dim));
  if (config_.use_layer_norm) {
    encoder_->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({config_.encoder_dim})));
  }
  encoder_->push_back(torch::nn::Functional(torch::relu));
  register_module("encoder", encoder_);

  const int fused_in = config_.encoder_dim + config_.workspace_dim;
  query_proj_ = register_module("query_proj", torch::nn::Linear(fused_in, config_.stm_key_dim));
  stm_context_proj_ = register_module(
      "stm_context_proj",
      torch::nn::Linear(config_.stm_value_dim, config_.controller_dim));
  ltm_query_proj_ = register_module("ltm_query_proj", torch::nn::Linear(fused_in, config_.ltm_dim));
  ltm_context_proj_ =
      register_module("ltm_context_proj", torch::nn::Linear(config_.ltm_dim, config_.controller_dim));
  ltm_write_proj_ =
      register_module("ltm_write_proj", torch::nn::Linear(config_.controller_dim, config_.ltm_slots));
  ltm_gate_proj_ =
      register_module("ltm_gate_proj", torch::nn::Linear(config_.controller_dim, config_.ltm_slots));
  gate_proj_ = register_module(
      "gate_proj",
      torch::nn::Linear(config_.controller_dim * 2 + config_.workspace_dim + config_.encoder_dim, 1));
  controller_proj_ = register_module(
      "controller_proj",
      torch::nn::Linear(config_.controller_dim * 2 + config_.workspace_dim + config_.encoder_dim, config_.controller_dim));
  workspace_cell_ = register_module(
      "workspace_cell",
      torch::nn::GRUCell(config_.encoder_dim + config_.controller_dim, config_.workspace_dim));
  stm_key_write_ = register_module(
      "stm_key_write",
      torch::nn::Linear(config_.workspace_dim + config_.encoder_dim, config_.stm_key_dim));
  stm_value_write_ = register_module(
      "stm_value_write",
      torch::nn::Linear(config_.workspace_dim + config_.encoder_dim, config_.stm_value_dim));

  const int head_in = config_.workspace_dim + config_.controller_dim + config_.encoder_dim;
  policy_head_ = register_module("policy_head", torch::nn::Linear(head_in, config_.action_dim));
  value_head_ = register_module("value_head", torch::nn::Linear(head_in, ppo_config_.value_num_atoms));
  next_goal_head_ = register_module("next_goal_head", torch::nn::Linear(head_in, 3));

  ltm_basis_keys_ = register_parameter(
      "ltm_basis_keys",
      torch::randn({config_.ltm_slots, config_.ltm_dim}) * 0.02);
  ltm_basis_values_ = register_parameter(
      "ltm_basis_values",
      torch::randn({config_.ltm_slots, config_.ltm_dim}) * 0.02);
  support_ = register_buffer(
      "value_support",
      torch::linspace(ppo_config_.value_v_min, ppo_config_.value_v_max, ppo_config_.value_num_atoms));
}

ContinuumState SharedActorCriticImpl::initial_state(std::int64_t batch_size, const torch::Device& device) const {
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto i64 = torch::TensorOptions().dtype(torch::kLong).device(device);
  return {
      torch::zeros({batch_size, config_.workspace_dim}, f32),
      torch::zeros({batch_size, config_.stm_slots, config_.stm_key_dim}, f32),
      torch::zeros({batch_size, config_.stm_slots, config_.stm_value_dim}, f32),
      torch::zeros({batch_size, config_.stm_slots}, f32),
      torch::zeros({batch_size}, i64),
      torch::zeros({batch_size, config_.ltm_slots}, f32),
      torch::zeros({batch_size}, i64),
  };
}

ContinuumState SharedActorCriticImpl::apply_episode_starts(ContinuumState state, torch::Tensor episode_starts) const {
  if (!episode_starts.defined()) {
    return state;
  }
  torch::Tensor mask = episode_starts.to(state.workspace.device()).to(torch::kFloat32).view({-1});
  state.workspace = maybe_zero_mask(state.workspace, mask);
  state.stm_keys = maybe_zero_mask(state.stm_keys, mask);
  state.stm_values = maybe_zero_mask(state.stm_values, mask);
  state.stm_strengths = maybe_zero_mask(state.stm_strengths, mask);
  state.stm_write_index = state.stm_write_index * (1 - mask.to(torch::kLong));
  state.ltm_coeffs = maybe_zero_mask(state.ltm_coeffs, mask);
  state.timestep = state.timestep * (1 - mask.to(torch::kLong));
  return state;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SharedActorCriticImpl::read_memories(
    const torch::Tensor& encoded,
    const ContinuumState& state) {
  const torch::Tensor query_input = torch::cat({encoded, state.workspace}, -1);
  const torch::Tensor stm_query = query_proj_->forward(query_input).unsqueeze(1);
  const torch::Tensor stm_scores = torch::bmm(stm_query, state.stm_keys.transpose(1, 2)).squeeze(1);
  const torch::Tensor stm_weights = masked_softmax(stm_scores, state.stm_strengths);
  const torch::Tensor stm_read = torch::bmm(stm_weights.unsqueeze(1), state.stm_values).squeeze(1);

  const torch::Tensor ltm_query = ltm_query_proj_->forward(query_input);
  const torch::Tensor ltm_scores =
      torch::matmul(ltm_query, ltm_basis_keys_.to(ltm_query.device()).transpose(0, 1)) + state.ltm_coeffs;
  const torch::Tensor ltm_weights = torch::softmax(ltm_scores, -1);
  const torch::Tensor ltm_read = torch::matmul(ltm_weights, ltm_basis_values_.to(ltm_query.device()));

  const torch::Tensor stm_ctx = stm_context_proj_->forward(stm_read);
  const torch::Tensor ltm_ctx = ltm_context_proj_->forward(ltm_read);
  return {stm_ctx, ltm_ctx, query_input};
}

ContinuumState SharedActorCriticImpl::write_short_term_memory(
    ContinuumState state,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  const torch::Tensor write_index = state.stm_write_index.remainder(config_.stm_slots);
  const torch::Tensor slot_mask = torch::one_hot(write_index, config_.stm_slots)
                                      .to(state.stm_keys.device())
                                      .to(torch::kFloat32);
  const torch::Tensor slot_mask_keys = slot_mask.unsqueeze(-1);
  state.stm_strengths = state.stm_strengths * config_.retired_decay;
  state.stm_keys = state.stm_keys * (1.0 - slot_mask_keys) + key.unsqueeze(1) * slot_mask_keys;
  state.stm_values = state.stm_values * (1.0 - slot_mask_keys) + value.unsqueeze(1) * slot_mask_keys;
  state.stm_strengths = state.stm_strengths * (1.0 - slot_mask) + slot_mask;
  state.stm_write_index = (state.stm_write_index + 1).remainder(config_.stm_slots);
  return state;
}

ContinuumState SharedActorCriticImpl::maybe_consolidate(
    ContinuumState state,
    const torch::Tensor& controller_hidden) {
  const torch::Tensor should_consolidate =
      state.timestep.remainder(config_.consolidation_stride).eq(0).to(torch::kFloat32).unsqueeze(-1);
  const torch::Tensor strengths = state.stm_strengths / state.stm_strengths.sum(-1, true).clamp_min(1.0e-6);
  const torch::Tensor stm_summary = torch::bmm(strengths.unsqueeze(1), state.stm_values).squeeze(1);
  const torch::Tensor write_logits = ltm_write_proj_->forward(controller_hidden + stm_context_proj_->forward(stm_summary));
  const torch::Tensor write_gate = torch::sigmoid(ltm_gate_proj_->forward(controller_hidden));
  const torch::Tensor delta = torch::tanh(write_logits) * write_gate * should_consolidate;
  state.ltm_coeffs = torch::clamp(state.ltm_coeffs + delta, -8.0, 8.0);
  state.stm_strengths = torch::where(should_consolidate.expand_as(state.stm_strengths) > 0.0,
                                     state.stm_strengths * config_.retired_decay,
                                     state.stm_strengths);
  return state;
}

PolicyOutput SharedActorCriticImpl::forward_step(torch::Tensor obs, ContinuumState state, torch::Tensor episode_starts) {
  state = apply_episode_starts(std::move(state), std::move(episode_starts));
  const torch::Tensor encoded = encoder_->forward(obs);
  const auto [stm_ctx, ltm_ctx, query_input] = read_memories(encoded, state);
  const torch::Tensor gate_input = torch::cat({encoded, state.workspace, stm_ctx, ltm_ctx}, -1);
  const torch::Tensor gate = torch::sigmoid(gate_proj_->forward(gate_input));
  const torch::Tensor fused_ctx = gate * stm_ctx + (1.0 - gate) * ltm_ctx;
  const torch::Tensor controller_hidden = torch::relu(controller_proj_->forward(gate_input));
  const torch::Tensor workspace_in = torch::cat({encoded, fused_ctx}, -1);
  const torch::Tensor new_workspace = workspace_cell_->forward(workspace_in, state.workspace);
  const torch::Tensor memory_in = torch::cat({encoded, new_workspace}, -1);
  const torch::Tensor new_key = torch::tanh(stm_key_write_->forward(memory_in));
  const torch::Tensor new_value = torch::tanh(stm_value_write_->forward(memory_in));
  state.workspace = new_workspace;
  state = write_short_term_memory(std::move(state), new_key, new_value);
  state = maybe_consolidate(std::move(state), controller_hidden);
  state.timestep = state.timestep + 1;

  const torch::Tensor head_input = torch::cat({encoded, fused_ctx, new_workspace}, -1);
  const torch::Tensor value_logits = value_head_->forward(head_input);
  return {
      policy_head_->forward(head_input),
      value_logits,
      expected_value(value_logits),
      sample_value(value_logits),
      next_goal_head_->forward(head_input),
      std::move(state),
  };
}

SequenceOutput SharedActorCriticImpl::forward_sequence(torch::Tensor obs_seq, ContinuumState state, torch::Tensor episode_starts) {
  const auto time = obs_seq.size(0);
  std::vector<torch::Tensor> policy_logits;
  std::vector<torch::Tensor> value_logits;
  std::vector<torch::Tensor> expected_values;
  std::vector<torch::Tensor> sampled_values;
  std::vector<torch::Tensor> next_goal_logits;
  policy_logits.reserve(time);
  value_logits.reserve(time);
  expected_values.reserve(time);
  sampled_values.reserve(time);
  next_goal_logits.reserve(time);

  for (std::int64_t t = 0; t < time; ++t) {
    torch::Tensor starts;
    if (episode_starts.defined()) {
      starts = episode_starts[t];
    }
    PolicyOutput out = forward_step(obs_seq[t], std::move(state), starts);
    policy_logits.push_back(out.policy_logits);
    value_logits.push_back(out.value_logits);
    expected_values.push_back(out.expected_values);
    sampled_values.push_back(out.sampled_values);
    next_goal_logits.push_back(out.next_goal_logits);
    state = std::move(out.state);
  }

  return {
      torch::stack(policy_logits, 0),
      torch::stack(value_logits, 0),
      torch::stack(expected_values, 0),
      torch::stack(sampled_values, 0),
      torch::stack(next_goal_logits, 0),
      std::move(state),
  };
}

torch::Tensor SharedActorCriticImpl::expected_value(torch::Tensor value_logits) const {
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  return (probs * support_.to(value_logits.device())).sum(-1);
}

torch::Tensor SharedActorCriticImpl::sample_value(torch::Tensor value_logits) const {
  const torch::Tensor indices = sample_categorical_from_logits(value_logits);
  return support_.to(value_logits.device()).index_select(0, indices);
}

torch::Tensor SharedActorCriticImpl::value_variance(torch::Tensor value_logits) const {
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  const torch::Tensor support = support_.to(value_logits.device());
  const torch::Tensor mean = (probs * support).sum(-1);
  const torch::Tensor second = (probs * support.pow(2)).sum(-1);
  return (second - mean.pow(2)).clamp_min(0.0);
}

torch::Tensor SharedActorCriticImpl::value_entropy(torch::Tensor value_logits) const {
  const torch::Tensor probs = torch::softmax(value_logits, -1);
  return -(probs * torch::log(probs + 1.0e-8)).sum(-1);
}

torch::Tensor SharedActorCriticImpl::support() const {
  return support_;
}

const ModelConfig& SharedActorCriticImpl::config() const {
  return config_;
}

const PPOConfig& SharedActorCriticImpl::ppo_config() const {
  return ppo_config_;
}

SharedActorCritic load_shared_model(const std::string& checkpoint_path, const std::string& device) {
  namespace fs = std::filesystem;

  const fs::path base(checkpoint_path);
  const ExperimentConfig config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_checkpoint_metadata(metadata, config);

  SharedActorCritic model(config.model, config.ppo);
  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string());
  model->load(archive);
  model->to(torch::Device(device));
  model->eval();
  return model;
}

SharedActorCritic clone_shared_model(const SharedActorCritic& source, const torch::Device& device) {
  torch::NoGradGuard no_grad;
  SharedActorCritic clone(source->config(), source->ppo_config());
  clone->to(device);
  auto dst_params = clone->named_parameters(true);
  for (const auto& item : source->named_parameters(true)) {
    dst_params[item.key()].copy_(item.value().to(device));
  }

  auto dst_buffers = clone->named_buffers(true);
  for (const auto& item : source->named_buffers(true)) {
    if (dst_buffers.find(item.key()) != nullptr) {
      dst_buffers[item.key()].copy_(item.value().to(device));
    }
  }

  if (source->is_training()) {
    clone->train();
  } else {
    clone->eval();
  }
  return clone;
}

}  // namespace pulsar

#endif
