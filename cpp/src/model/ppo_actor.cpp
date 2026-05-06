#include "pulsar/model/ppo_actor.hpp"

#ifdef PULSAR_HAS_TORCH

#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"

namespace pulsar {
namespace {

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

void append_encoder_block(
    torch::nn::Sequential& encoder,
    int in_dim,
    int out_dim,
    bool use_layer_norm) {
  encoder->push_back(torch::nn::Linear(in_dim, out_dim));
  if (use_layer_norm) {
    encoder->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({out_dim})));
  }
  encoder->push_back(torch::nn::Functional(torch::relu));
}

void validate_model_config(const ModelConfig& config) {
  auto require_positive = [](int value, const char* field) {
    if (value <= 0) {
      throw std::invalid_argument(std::string("ModelConfig.") + field + " must be positive.");
    }
  };

  require_positive(config.observation_dim, "observation_dim");
  require_positive(config.action_dim, "action_dim");
  require_positive(config.encoder_dim, "encoder_dim");
  require_positive(config.workspace_dim, "workspace_dim");
  require_positive(config.stm_slots, "stm_slots");
  require_positive(config.stm_key_dim, "stm_key_dim");
  require_positive(config.stm_value_dim, "stm_value_dim");
  require_positive(config.ltm_slots, "ltm_slots");
  require_positive(config.ltm_dim, "ltm_dim");
  require_positive(config.controller_dim, "controller_dim");
  require_positive(config.consolidation_stride, "consolidation_stride");
  require_positive(config.value_hidden_dim, "value_hidden_dim");
  require_positive(config.value_num_atoms, "value_num_atoms");

  if (config.value_num_atoms < 2) {
    throw std::invalid_argument("ModelConfig.value_num_atoms must be >= 2 for distributional RL.");
  }
  if (!(config.value_v_max > config.value_v_min)) {
    throw std::invalid_argument("ModelConfig.value_v_max must be greater than value_v_min.");
  }
}

}  // namespace

LoRALinearImpl::LoRALinearImpl(int in_features, int out_features, int rank, float lora_alpha)
    : rank_(rank), scale_(lora_alpha / static_cast<float>(rank)) {
  base = register_module("base", torch::nn::Linear(in_features, out_features));

  A = register_parameter(
      "A",
      torch::randn({rank, in_features}) * 0.02F);
  B = register_parameter(
      "B",
      torch::zeros({out_features, rank}));
}

torch::Tensor LoRALinearImpl::forward(torch::Tensor x) {
  torch::Tensor base_out = base->forward(x);
  torch::Tensor lora_out = scale_ * torch::matmul(
      torch::matmul(x, A.transpose(0, 1)),
      B.transpose(0, 1));
  return base_out + lora_out;
}

torch::Tensor LoRALinearImpl::forward_eggroll_population(
    torch::Tensor x,
    const torch::Tensor& A_stack,
    const torch::Tensor& B_stack,
    float sigma) {
  const auto population = A_stack.size(0);
  if (population <= 0 || x.size(0) % population != 0) {
    throw std::invalid_argument("LoRALinearImpl::forward_eggroll_population received incompatible population dimensions.");
  }
  const auto member_batch = x.size(0) / population;
  torch::Tensor base_out = forward(x).view({population, member_batch, out_features()});
  torch::Tensor x_view = x.view({population, member_batch, in_features()});
  torch::Tensor low_rank = torch::bmm(
      torch::bmm(x_view, A_stack.transpose(1, 2)),
      B_stack.transpose(1, 2));
  return (base_out + low_rank * sigma).view({x.size(0), out_features()});
}

void LoRALinearImpl::reset_lora_parameters() {
  A.normal_(0.0, 0.02);
  B.zero_();
}

std::vector<torch::Tensor> LoRALinearImpl::lora_parameters() const {
  return {A, B};
}

std::vector<torch::Tensor> LoRALinearImpl::lora_parameters_flat() const {
  return {A.view({-1}), B.view({-1})};
}

void LoRALinearImpl::restore_lora_parameters(const std::vector<torch::Tensor>& params) {
  torch::NoGradGuard no_grad;
  A.copy_(params[0].view_as(A));
  B.copy_(params[1].view_as(B));
}

void LoRALinearImpl::apply_base_weight_update(const torch::Tensor& delta_weight) {
  torch::NoGradGuard no_grad;
  base->weight.add_(delta_weight.to(base->weight.device()).to(base->weight.dtype()));
}

int LoRALinearImpl::in_features() const {
  return static_cast<int>(base->weight.size(1));
}

int LoRALinearImpl::out_features() const {
  return static_cast<int>(base->weight.size(0));
}

int LoRALinearImpl::rank() const {
  return rank_;
}

float LoRALinearImpl::scale() const {
  return scale_;
}

GoalCriticImpl::GoalCriticImpl(int feature_dim, int action_dim, int num_atoms, int hidden_dim)
    : action_dim_(action_dim), hidden_dim_(hidden_dim) {
  const int input_dim = feature_dim + action_dim + 1;
  input_proj_ = register_module(
      "input_proj", torch::nn::Linear(input_dim, hidden_dim));
  output_proj_ = register_module(
      "output_proj", torch::nn::Linear(hidden_dim, num_atoms));
}

torch::Tensor GoalCriticImpl::forward(
    const torch::Tensor& features,
    const torch::Tensor& action_ids,
    const torch::Tensor& goal_value) {
  torch::Tensor action_one_hot = torch::nn::functional::one_hot(
      action_ids.to(torch::kLong), action_dim_).to(features.device()).to(torch::kFloat32);
  torch::Tensor goal_expanded = goal_value.unsqueeze(-1);
  torch::Tensor combined = torch::cat({features, action_one_hot, goal_expanded}, -1);
  torch::Tensor hidden = torch::relu(input_proj_->forward(combined));
  return output_proj_->forward(hidden);
}

torch::nn::Sequential PPOActorImpl::make_value_win_head(int input_dim) const {
  torch::nn::Sequential head;
  head->push_back(torch::nn::Linear(input_dim, config_.value_hidden_dim));
  head->push_back(torch::nn::Functional(torch::relu));
  head->push_back(torch::nn::Linear(config_.value_hidden_dim, config_.value_num_atoms));
  return head;
}

PPOActorImpl::PPOActorImpl(ModelConfig config, const GoalCriticConfig& goal_critic_config)
    : config_(std::move(config)), goal_critic_config_(goal_critic_config) {
  validate_model_config(config_);
  append_encoder_block(encoder_, config_.observation_dim, config_.encoder_dim, config_.use_layer_norm);
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

  feature_dim_ = config_.workspace_dim + config_.controller_dim + config_.encoder_dim;
  {
    if (config_.policy_hidden_dim > 0) {
      policy_hidden_ = torch::nn::Sequential();
      policy_hidden_->push_back(torch::nn::Linear(feature_dim_, config_.policy_hidden_dim));
      policy_hidden_->push_back(torch::nn::Functional(torch::relu));
      register_module("policy_hidden", policy_hidden_);
      policy_lora_ = LoRALinear(
          config_.policy_hidden_dim, config_.action_dim, 4, 8.0F);
    } else {
      policy_lora_ = LoRALinear(feature_dim_, config_.action_dim, 4, 8.0F);
    }
    register_module("policy_lora", policy_lora_);
  }

  value_head_win_ = make_value_win_head(feature_dim_);
  register_module("value_head_win", value_head_win_);

  {
    const float atom_delta = (config_.value_v_max - config_.value_v_min)
        / static_cast<float>(config_.value_num_atoms - 1);
    atom_support_win_ = register_buffer(
        "atom_support_win",
        torch::arange(static_cast<float>(config_.value_num_atoms),
                      torch::TensorOptions().dtype(torch::kFloat32))
            .mul_(atom_delta)
            .add_(config_.value_v_min));
  }

  {
    int goal_num_atoms = goal_critic_config_.num_atoms;
    if (goal_num_atoms <= 0) {
      goal_num_atoms = 51;
    }
    goal_critic_ = GoalCritic(
        feature_dim_, config_.action_dim, goal_num_atoms, config_.value_hidden_dim);
    register_module("goal_critic", goal_critic_);

    float g_v_min = goal_critic_config_.v_min;
    float g_v_max = compute_goal_critic_v_max(goal_critic_config_);
    float g_atom_delta = (g_v_max - g_v_min) / static_cast<float>(goal_num_atoms - 1);
    atom_support_goal_ = register_buffer(
        "atom_support_goal",
        torch::arange(static_cast<float>(goal_num_atoms),
                      torch::TensorOptions().dtype(torch::kFloat32))
            .mul_(g_atom_delta)
            .add_(g_v_min));
  }

  ltm_basis_keys_ = register_parameter(
      "ltm_basis_keys",
      torch::randn({config_.ltm_slots, config_.ltm_dim}) * 0.02);
  ltm_basis_values_ = register_parameter(
      "ltm_basis_values",
      torch::randn({config_.ltm_slots, config_.ltm_dim}) * 0.02);
}

ContinuumState PPOActorImpl::initial_state(std::int64_t batch_size, const torch::Device& device) const {
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

ContinuumState PPOActorImpl::apply_episode_starts(ContinuumState state, torch::Tensor episode_starts) const {
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPOActorImpl::read_memories(
    const torch::Tensor& encoded,
    const ContinuumState& state) {
  const torch::Tensor query_input = torch::cat({encoded, state.workspace}, -1);
  const torch::Tensor stm_query = query_proj_->forward(query_input).unsqueeze(1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(config_.stm_key_dim));
  const torch::Tensor stm_scores = torch::bmm(stm_query, state.stm_keys.transpose(1, 2)).squeeze(1) * scale;
  const torch::Tensor stm_weights = masked_softmax(stm_scores, state.stm_strengths);
  const torch::Tensor stm_read = torch::bmm(stm_weights.unsqueeze(1), state.stm_values).squeeze(1);

  const torch::Tensor ltm_query = ltm_query_proj_->forward(query_input);
  const torch::Tensor ltm_scores = torch::matmul(ltm_query, ltm_basis_keys_.transpose(0, 1)) + state.ltm_coeffs;
  const torch::Tensor ltm_weights = torch::softmax(ltm_scores, -1);
  const torch::Tensor ltm_read = torch::matmul(ltm_weights, ltm_basis_values_);

  const torch::Tensor stm_ctx = stm_context_proj_->forward(stm_read);
  const torch::Tensor ltm_ctx = ltm_context_proj_->forward(ltm_read);
  return {stm_ctx, ltm_ctx, query_input};
}

ContinuumState PPOActorImpl::write_short_term_memory(
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

ContinuumState PPOActorImpl::maybe_consolidate(
    ContinuumState state,
    const torch::Tensor& controller_hidden) {
  const torch::Tensor should_consolidate =
      (state.timestep > 0).logical_and_(
          state.timestep.remainder(config_.consolidation_stride).eq(0))
          .to(torch::kFloat32).unsqueeze(-1);
  const torch::Tensor strengths = state.stm_strengths / state.stm_strengths.sum(-1, true).clamp_min(1.0e-6);
  const torch::Tensor stm_summary = torch::bmm(strengths.unsqueeze(1), state.stm_values).squeeze(1);
  const torch::Tensor write_logits = ltm_write_proj_->forward(controller_hidden + stm_context_proj_->forward(stm_summary));
  const torch::Tensor write_gate = torch::sigmoid(ltm_gate_proj_->forward(controller_hidden));
  const torch::Tensor delta = torch::tanh(write_logits) * write_gate * should_consolidate;
  state.ltm_coeffs = torch::clamp(state.ltm_coeffs + delta, -8.0, 8.0);
  return state;
}

ActorStepOutput PPOActorImpl::forward_encoded_step(
    torch::Tensor encoded,
    ContinuumState state,
    torch::Tensor episode_starts) {
  state = apply_episode_starts(std::move(state), std::move(episode_starts));
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

  const torch::Tensor features = torch::cat({encoded, fused_ctx, new_workspace}, -1);

  torch::Tensor policy_logits;
  if (!policy_hidden_.is_empty()) {
    policy_logits = policy_lora_->forward(policy_hidden_->forward(features));
  } else {
    policy_logits = policy_lora_->forward(features);
  }

  return {
      policy_logits,
      encoded,
      value_head_win_->forward(features),
      torch::Tensor{},
      features,
      std::move(state),
  };
}

ActorStepOutput PPOActorImpl::forward_step(torch::Tensor obs, ContinuumState state, torch::Tensor episode_starts) {
  return forward_encoded_step(encoder_->forward(obs), std::move(state), std::move(episode_starts));
}

ActorSequenceOutput PPOActorImpl::forward_sequence(
    torch::Tensor obs_seq,
    ContinuumState state,
    torch::Tensor episode_starts) {
  const auto time = obs_seq.size(0);
  const auto batch = obs_seq.size(1);
  const torch::Tensor encoded_seq =
      encoder_->forward(obs_seq.reshape({time * batch, config_.observation_dim}))
          .reshape({time, batch, config_.encoder_dim});
  std::vector<torch::Tensor> policy_logits;
  std::vector<torch::Tensor> encoded_seq_stack;
  std::vector<torch::Tensor> value_win_logits;
  std::vector<torch::Tensor> features;
  policy_logits.reserve(time);
  encoded_seq_stack.reserve(time);
  value_win_logits.reserve(time);
  features.reserve(time);

  for (std::int64_t t = 0; t < time; ++t) {
    torch::Tensor starts;
    if (episode_starts.defined()) {
      starts = episode_starts[t];
    }
    ActorStepOutput out = forward_encoded_step(encoded_seq[t], std::move(state), starts);
    policy_logits.push_back(out.policy_logits);
    encoded_seq_stack.push_back(out.encoded);
    value_win_logits.push_back(out.value_win_logits);
    features.push_back(out.features);
    state = std::move(out.state);
  }

  return {
      torch::stack(policy_logits, 0),
      torch::stack(encoded_seq_stack, 0),
      torch::stack(value_win_logits, 0),
      torch::Tensor{},
      torch::stack(features, 0),
      std::move(state),
  };
}

torch::Tensor PPOActorImpl::value_win_support() const {
  return atom_support_win_;
}

torch::Tensor PPOActorImpl::goal_critic_support() const {
  return atom_support_goal_;
}

int PPOActorImpl::feature_dim() const {
  return feature_dim_;
}

const ModelConfig& PPOActorImpl::config() const {
  return config_;
}

const GoalCriticConfig& PPOActorImpl::goal_critic_config() const {
  return goal_critic_config_;
}

std::vector<torch::Tensor> PPOActorImpl::es_lora_parameters() const {
  return policy_lora_->lora_parameters();
}

std::vector<torch::Tensor> PPOActorImpl::es_lora_parameters_flat() const {
  return policy_lora_->lora_parameters_flat();
}

void PPOActorImpl::restore_es_lora_parameters(const std::vector<torch::Tensor>& params) {
  policy_lora_->restore_lora_parameters(params);
}

void PPOActorImpl::apply_lora_perturbation(
    const std::vector<torch::Tensor>& perturbation, float sigma) {
  torch::NoGradGuard no_grad;
  auto params = es_lora_parameters();
  for (std::size_t i = 0; i < params.size(); ++i) {
    params[i].add_(perturbation[i], sigma);
  }
}

torch::Tensor PPOActorImpl::policy_eggroll_logits(
    const torch::Tensor& features,
    const torch::Tensor& A_stack,
    const torch::Tensor& B_stack,
    float sigma) {
  torch::Tensor policy_input = features;
  if (!policy_hidden_.is_empty()) {
    policy_input = policy_hidden_->forward(policy_input);
  }
  return policy_lora_->forward_eggroll_population(policy_input, A_stack, B_stack, sigma);
}

void PPOActorImpl::apply_policy_eggroll_update(const torch::Tensor& delta_weight) {
  policy_lora_->apply_base_weight_update(delta_weight);
}

const LoRALinear& PPOActorImpl::policy_lora() const {
  return policy_lora_;
}

GoalCritic& PPOActorImpl::goal_critic() {
  return goal_critic_;
}

std::vector<std::string> PPOActorImpl::enabled_critic_heads() const {
  return {"extrinsic"};
}

PPOActor load_ppo_actor(const std::string& checkpoint_path, const std::string& device) {
  namespace fs = std::filesystem;
  const fs::path base(checkpoint_path);
  const ExperimentConfig config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_inference_checkpoint_metadata(metadata, config);
  if (metadata.architecture_name != "ppo_continuum"
      && metadata.architecture_name != "dappo_continuum"
      && metadata.architecture_name != "continuum_goal_conditioned"
      && metadata.architecture_name != "policy_snapshot") {
    throw std::runtime_error("Checkpoint is not a continuum actor checkpoint.");
  }

  torch::Device torch_device(device);
  auto model = PPOActor(config.model, config.goal_critic);
  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string(), torch_device);
  model->load(archive);
  model->to(torch_device);
  model->eval();
  return model;
}

PPOActor clone_ppo_actor(const PPOActor& source, const torch::Device& device) {
  if (!source) {
    return nullptr;
  }
  auto clone = PPOActor(source->config(), source->goal_critic_config());
  torch::serialize::OutputArchive out;
  source->save(out);
  torch::serialize::InputArchive in;
  std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
  out.save_to(buffer);
  in.load_from(buffer, device);
  clone->load(in);
  clone->to(device);
  return clone;
}

}  // namespace pulsar

#endif
