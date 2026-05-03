#include "pulsar/model/ppo_actor.hpp"

#ifdef PULSAR_HAS_TORCH

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

  // Distributional RL requires at least 2 atoms (divides by num_atoms - 1).
  if (config.value_num_atoms < 2) {
    throw std::invalid_argument("ModelConfig.value_num_atoms must be >= 2 for distributional RL.");
  }
  if (!(config.value_v_max > config.value_v_min)) {
    throw std::invalid_argument("ModelConfig.value_v_max must be greater than value_v_min.");
  }
}

}  // namespace

torch::nn::Sequential PPOActorImpl::make_value_head(int input_dim, const CriticHeadConfig& head_config) const {
  torch::nn::Sequential head;
  head->push_back(torch::nn::Linear(input_dim, head_config.value_hidden_dim));
  head->push_back(torch::nn::Functional(torch::relu));
  head->push_back(torch::nn::Linear(head_config.value_hidden_dim, head_config.value_num_atoms));
  return head;
}

torch::nn::Sequential PPOActorImpl::make_forward_head(int input_dim, int forward_action_dim) const {
  const int hidden = config_.forward_model.hidden_dim;
  torch::nn::Sequential head;
  head->push_back(torch::nn::Linear(input_dim + forward_action_dim, hidden));
  head->push_back(torch::nn::Functional(torch::relu));
  for (int i = 1; i < config_.forward_model.num_layers; ++i) {
    head->push_back(torch::nn::Linear(hidden, hidden));
    head->push_back(torch::nn::Functional(torch::relu));
  }
  head->push_back(torch::nn::Linear(hidden, config_.encoder_dim));
  return head;
}

torch::nn::Sequential PPOActorImpl::make_inverse_head() const {
  const int hidden = config_.inverse_model.hidden_dim;
  torch::nn::Sequential head;
  head->push_back(torch::nn::Linear(config_.encoder_dim * 2, hidden));
  head->push_back(torch::nn::Functional(torch::relu));
  for (int i = 1; i < config_.inverse_model.num_layers; ++i) {
    head->push_back(torch::nn::Linear(hidden, hidden));
    head->push_back(torch::nn::Functional(torch::relu));
  }
  head->push_back(torch::nn::Linear(hidden, config_.action_dim));
  return head;
}

torch::Tensor PPOActorImpl::make_atom_support(float v_min, float v_max, int num_atoms) const {
  const float atom_delta = (v_max - v_min) / static_cast<float>(num_atoms - 1);
  return torch::arange(
      static_cast<float>(num_atoms),
      torch::TensorOptions().dtype(torch::kFloat32))
      .mul_(atom_delta)
      .add_(v_min);
}

void PPOActorImpl::build_value_head(
    const std::string& name,
    torch::nn::Sequential& head,
    torch::Tensor& support,
    const CriticHeadConfig& head_cfg) {
  head = make_value_head(feature_dim_, head_cfg);
  register_module("value_head_" + name, head);
  support = register_buffer(
      "atom_support_" + name,
      make_atom_support(head_cfg.value_v_min, head_cfg.value_v_max, head_cfg.value_num_atoms));
  value_heads_[name] = &head;
  atom_supports_[name] = &support;
  enabled_critic_heads_.push_back(name);
}

PPOActorImpl::PPOActorImpl(ModelConfig config)
    : config_(std::move(config)) {
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
  policy_head_ = register_module("policy_head", torch::nn::Linear(feature_dim_, config_.action_dim));

  const CriticHeadConfig default_head{
      true, config_.value_hidden_dim, config_.value_num_atoms,
      config_.value_v_min, config_.value_v_max};

  // NOTE: The actor currently uses ModelConfig defaults for value-head dimensions.
  // Full per-head CriticConfig (v_min, v_max, num_atoms, hidden_dim) from
  // ExperimentConfig is not yet wired through the PPOActorImpl constructor.
  // Only the enabled/disabled flag from CriticConfig is respected at the call site
  // (see load_ppo_actor / APPOTrainer).  For now all heads are built unconditionally;
  // heads that are not trained simply receive zero loss weight via head_weights_.
  build_value_head("extrinsic", value_head_ext_, atom_support_ext_, default_head);
  build_value_head("curiosity", value_head_cur_, atom_support_cur_, default_head);
  build_value_head("learning_progress", value_head_learn_, atom_support_learn_, default_head);

  CriticHeadConfig ctrl_head_cfg = default_head;
  ctrl_head_cfg.enabled = false;  // Controllability head is auxiliary only.
  build_value_head("controllability", value_head_ctrl_, atom_support_ctrl_, ctrl_head_cfg);

  forward_head_ = make_forward_head(config_.encoder_dim, config_.action_dim);
  register_module("forward_head", forward_head_);

  inverse_head_ = make_inverse_head();
  register_module("inverse_head", inverse_head_);

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
  const torch::Tensor stm_scores = torch::bmm(stm_query, state.stm_keys.transpose(1, 2)).squeeze(1);
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
      state.timestep.remainder(config_.consolidation_stride).eq(0).to(torch::kFloat32).unsqueeze(-1);
  const torch::Tensor strengths = state.stm_strengths / state.stm_strengths.sum(-1, true).clamp_min(1.0e-6);
  const torch::Tensor stm_summary = torch::bmm(strengths.unsqueeze(1), state.stm_values).squeeze(1);
  const torch::Tensor write_logits = ltm_write_proj_->forward(controller_hidden + stm_context_proj_->forward(stm_summary));
  const torch::Tensor write_gate = torch::sigmoid(ltm_gate_proj_->forward(controller_hidden));
  const torch::Tensor delta = torch::tanh(write_logits) * write_gate * should_consolidate;
  state.ltm_coeffs = torch::clamp(state.ltm_coeffs + delta, -8.0, 8.0);
  state.stm_strengths = torch::where(
      should_consolidate.expand_as(state.stm_strengths) > 0.0,
      state.stm_strengths * config_.retired_decay,
      state.stm_strengths);
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
  return {
      policy_head_->forward(features),
      encoded,
      {value_head_ext_->forward(features), atom_support_ext_},
      {value_head_cur_->forward(features), atom_support_cur_},
      {value_head_learn_->forward(features), atom_support_learn_},
      {value_head_ctrl_->forward(features), atom_support_ctrl_},
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
  std::vector<torch::Tensor> value_logits_ext;
  std::vector<torch::Tensor> value_logits_cur;
  std::vector<torch::Tensor> value_logits_learn;
  std::vector<torch::Tensor> value_logits_ctrl;
  std::vector<torch::Tensor> features;
  policy_logits.reserve(time);
  encoded_seq_stack.reserve(time);
  value_logits_ext.reserve(time);
  value_logits_cur.reserve(time);
  value_logits_learn.reserve(time);
  value_logits_ctrl.reserve(time);
  features.reserve(time);

  for (std::int64_t t = 0; t < time; ++t) {
    torch::Tensor starts;
    if (episode_starts.defined()) {
      starts = episode_starts[t];
    }
    ActorStepOutput out = forward_encoded_step(encoded_seq[t], std::move(state), starts);
    policy_logits.push_back(out.policy_logits);
    encoded_seq_stack.push_back(out.encoded);
    value_logits_ext.push_back(out.value_ext.logits);
    value_logits_cur.push_back(out.value_cur.logits);
    value_logits_learn.push_back(out.value_learn.logits);
    value_logits_ctrl.push_back(out.value_ctrl.logits);
    features.push_back(out.features);
    state = std::move(out.state);
  }

  return {
      torch::stack(policy_logits, 0),
      torch::stack(encoded_seq_stack, 0),
      {torch::stack(value_logits_ext, 0), atom_support_ext_},
      {torch::stack(value_logits_cur, 0), atom_support_cur_},
      {torch::stack(value_logits_learn, 0), atom_support_learn_},
      {torch::stack(value_logits_ctrl, 0), atom_support_ctrl_},
      torch::stack(features, 0),
      std::move(state),
  };
}

torch::Tensor PPOActorImpl::value_support(const std::string& head_name) const {
  auto it = atom_supports_.find(head_name);
  if (it != atom_supports_.end()) {
    return *(it->second);
  }
  return atom_support_ext_;
}

const ValueHeadOutput& PPOActorImpl::value_head_output(const std::string& head_name, const ActorStepOutput& output) const {
  if (head_name == "extrinsic") return output.value_ext;
  if (head_name == "curiosity") return output.value_cur;
  if (head_name == "learning_progress") return output.value_learn;
  if (head_name == "controllability") return output.value_ctrl;
  return output.value_ext;
}

const ValueHeadOutput& PPOActorImpl::value_head_output(const std::string& head_name, const ActorSequenceOutput& output) const {
  if (head_name == "extrinsic") return output.value_ext;
  if (head_name == "curiosity") return output.value_cur;
  if (head_name == "learning_progress") return output.value_learn;
  if (head_name == "controllability") return output.value_ctrl;
  return output.value_ext;
}

std::vector<std::string> PPOActorImpl::enabled_critic_heads() const {
  return enabled_critic_heads_;
}

int PPOActorImpl::feature_dim() const {
  return feature_dim_;
}

const ModelConfig& PPOActorImpl::config() const {
  return config_;
}

torch::Tensor PPOActorImpl::forward_predict_next(torch::Tensor encoded, torch::Tensor actions) {
  torch::Tensor action_features = torch::nn::functional::one_hot(
      actions.to(torch::kLong), config_.action_dim).to(encoded.device()).to(torch::kFloat32);
  torch::Tensor combined = torch::cat({encoded, action_features}, -1);
  return forward_head_->forward(combined);
}

torch::Tensor PPOActorImpl::forward_predict_action(torch::Tensor encoded_t, torch::Tensor encoded_tp1) {
  torch::Tensor combined = torch::cat({encoded_t, encoded_tp1}, -1);
  return inverse_head_->forward(combined);
}

torch::Tensor PPOActorImpl::compute_forward_prediction_error(
    torch::Tensor encoded, torch::Tensor actions, torch::Tensor encoded_tp1) {
  torch::Tensor predicted_next = forward_predict_next(encoded, actions);
  return torch::mse_loss(predicted_next, encoded_tp1, torch::Reduction::None).mean(-1);
}

PPOActor load_ppo_actor(const std::string& checkpoint_path, const std::string& device) {
  namespace fs = std::filesystem;
  const fs::path base(checkpoint_path);
  const ExperimentConfig config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_inference_checkpoint_metadata(metadata, config);
  if (metadata.architecture_name != "ppo_continuum" && metadata.architecture_name != "dappo_continuum") {
    throw std::runtime_error("Checkpoint is not a PPO actor checkpoint.");
  }

  torch::Device torch_device(device);
  auto model = PPOActor(config.model);
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
  auto clone = PPOActor(source->config());
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
