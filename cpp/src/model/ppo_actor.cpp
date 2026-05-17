#include "pulsar/model/ppo_actor.hpp"

#ifdef PULSAR_HAS_TORCH

#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/tracing/tracing.hpp"

namespace pulsar {
namespace {

void copy_module_tensors_to(const PPOActor& source, PPOActor& target, const torch::Device&) {
  torch::NoGradGuard no_grad;

  const auto source_params = source->named_parameters(true);
  auto target_params = target->named_parameters(true);
  for (const auto& item : source_params) {
    torch::Tensor* target_tensor = target_params.find(item.key());
    if (target_tensor == nullptr) {
      throw std::runtime_error("Missing cloned actor parameter: " + std::string(item.key()));
    }
    target_tensor->copy_(item.value(), /*non_blocking=*/false);
  }

  const auto source_buffers = source->named_buffers(true);
  auto target_buffers = target->named_buffers(true);
  for (const auto& item : source_buffers) {
    torch::Tensor* target_tensor = target_buffers.find(item.key());
    if (target_tensor == nullptr) {
      throw std::runtime_error("Missing cloned actor buffer: " + std::string(item.key()));
    }
    target_tensor->copy_(item.value(), /*non_blocking=*/false);
  }
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
  require_positive(config.num_encoder_blocks, "num_encoder_blocks");
  require_positive(config.sequence_length, "sequence_length");
  if (config.max_forward_samples < 0) {
    throw std::invalid_argument("ModelConfig.max_forward_samples must be non-negative (0 = unlimited).");
  }
  require_positive(config.value_hidden_dim, "value_hidden_dim");
}

void validate_es_lora_config(const ESLoraConfig& config) {
  if (config.rank <= 0) {
    throw std::invalid_argument("ESLoraConfig.rank must be positive.");
  }
  if (config.lora_alpha <= 0.0F) {
    throw std::invalid_argument("ESLoraConfig.lora_alpha must be positive.");
  }
}

}  // namespace

LoRALinearImpl::LoRALinearImpl(int in_features, int out_features, int rank, float lora_alpha)
    : rank_(rank), scale_(lora_alpha / static_cast<float>(rank) / 2.0F) {
  base = register_module("base", torch::nn::Linear(in_features, out_features));

  A = register_parameter(
      "A",
      torch::randn({rank, in_features}) * 0.01F);
  B = register_parameter(
      "B",
      torch::randn({out_features, rank}) * 0.01F);
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

  torch::Tensor x_A_stack = torch::bmm(x_view, A_stack.transpose(1, 2));
  torch::Tensor B_T_expanded = B.transpose(0, 1).unsqueeze(0).expand({population, -1, -1});
  torch::Tensor cross1 = torch::bmm(x_A_stack, B_T_expanded);

  torch::Tensor A_T_expanded = A.transpose(0, 1).unsqueeze(0).expand({population, -1, -1});
  torch::Tensor x_A = torch::bmm(x_view, A_T_expanded);
  torch::Tensor cross2 = torch::bmm(x_A, B_stack.transpose(1, 2));

  torch::Tensor pert_only = torch::bmm(x_A_stack, B_stack.transpose(1, 2));

  float s = scale_;
  return (base_out + s * sigma * cross1 + s * sigma * cross2 + s * sigma * sigma * pert_only)
      .view({x.size(0), out_features()});
}

void LoRALinearImpl::reset_lora_parameters() {
  torch::NoGradGuard no_grad;
  A.normal_(0.0, 0.01);
  B.normal_(0.0, 0.01);
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

Mamba2BlockImpl::Mamba2BlockImpl(int embed_dim, int sequence_length, bool use_layer_norm)
    : embed_dim_(embed_dim), sequence_length_(sequence_length), use_layer_norm_(use_layer_norm) {
  if (embed_dim_ <= 0) {
    throw std::invalid_argument("Mamba2Block requires positive embed_dim.");
  }
  if (sequence_length_ <= 0) {
    throw std::invalid_argument("Mamba2Block requires positive sequence_length.");
  }
  if (use_layer_norm_) {
    input_norm_ = register_module("input_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})));
    output_norm_ = register_module("output_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})));
  }
  causal_conv_ = register_module(
      "causal_conv",
      torch::nn::Conv1d(torch::nn::Conv1dOptions(embed_dim_, embed_dim_, 3)
                            .groups(embed_dim_)
                            .padding(2)));
  input_projection_ = register_module("input_projection", torch::nn::Linear(embed_dim_, 5 * embed_dim_));
  output_projection_ = register_module("output_projection", torch::nn::Linear(embed_dim_, embed_dim_));
  decay_bias_ = register_parameter("decay_bias", torch::full({embed_dim_}, 2.0F));
  skip_ = register_parameter("skip", torch::ones({embed_dim_}));
}

torch::Tensor Mamba2BlockImpl::forward(const torch::Tensor& tokens, const torch::Tensor& reset_mask) {
  PULSAR_TRACE_SCOPE_CAT("actor", "mamba2_block");
  const auto batch = tokens.size(0);
  const auto sequence = tokens.size(1);
  torch::Tensor block_input = use_layer_norm_ ? input_norm_->forward(tokens) : tokens;

  const torch::Tensor reset = reset_mask.defined()
      ? reset_mask.to(tokens.device()).to(tokens.scalar_type())
      : torch::Tensor{};

  torch::Tensor conv_out;
  if (reset.defined()) {
    const torch::Tensor weight = causal_conv_->weight.squeeze(1);
    const torch::Tensor zero_step = torch::zeros({batch, 1, embed_dim_}, tokens.options());
    torch::Tensor prev_1 = sequence > 1
        ? torch::cat({zero_step, block_input.slice(1, 0, sequence - 1)}, 1)
        : torch::zeros_like(block_input);
    torch::Tensor prev_2 = sequence > 2
        ? torch::cat({zero_step, zero_step, block_input.slice(1, 0, sequence - 2)}, 1)
        : torch::zeros_like(block_input);
    const torch::Tensor keep_prev_1 = (1.0F - reset).unsqueeze(-1);
    torch::Tensor previous_reset = torch::zeros_like(reset);
    if (sequence > 1) {
      previous_reset.slice(1, 1).copy_(reset.slice(1, 0, sequence - 1));
    }
    const torch::Tensor keep_prev_2 = ((1.0F - reset) * (1.0F - previous_reset)).unsqueeze(-1);
    prev_1 = prev_1 * keep_prev_1;
    prev_2 = prev_2 * keep_prev_2;
    conv_out =
        prev_2 * weight.select(1, 0).view({1, 1, embed_dim_})
        + prev_1 * weight.select(1, 1).view({1, 1, embed_dim_})
        + block_input * weight.select(1, 2).view({1, 1, embed_dim_});
    if (causal_conv_->bias.defined()) {
      conv_out = conv_out + causal_conv_->bias.view({1, 1, embed_dim_});
    }
  } else {
    // Local causal mixing before the selective scan. Conv1d expects [B, C, S].
    torch::Tensor conv_input = block_input.transpose(1, 2);
    conv_out = causal_conv_->forward(conv_input).narrow(2, 0, sequence).transpose(1, 2);
  }
  conv_out = torch::silu(conv_out);

  const auto projected = input_projection_->forward(conv_out).chunk(5, -1);
  const torch::Tensor x = torch::silu(projected[0]);
  const torch::Tensor b = torch::sigmoid(projected[1]);
  const torch::Tensor c = torch::sigmoid(projected[2]);
  const torch::Tensor z = torch::silu(projected[3]);
  const torch::Tensor retention = torch::sigmoid(projected[4] + decay_bias_.view({1, 1, embed_dim_}))
                                      .clamp(0.01, 0.9999);
  const torch::Tensor recurrent_input = b * x;

  torch::Tensor state = torch::zeros({batch, embed_dim_}, tokens.options());
  std::vector<torch::Tensor> states;
  states.reserve(static_cast<std::size_t>(sequence));
  for (int64_t t = 0; t < sequence; ++t) {
    if (reset.defined()) {
      const torch::Tensor keep = (1.0F - reset.select(1, t)).view({batch, 1});
      state = state * keep;
    }
    state = retention.select(1, t) * state + recurrent_input.select(1, t);
    states.push_back(state);
  }
  const torch::Tensor scanned = torch::stack(states, 1);

  torch::Tensor mixed = (c * scanned + skip_.view({1, 1, embed_dim_}) * x) * z;
  if (use_layer_norm_) {
    mixed = output_norm_->forward(mixed);
  }
  return tokens + output_projection_->forward(mixed);
}

torch::Tensor Mamba2BlockImpl::forward_step(
    const torch::Tensor& token,
    const torch::Tensor& previous_conv_2,
    const torch::Tensor& previous_conv_1,
    const torch::Tensor& previous_scan,
    torch::Tensor* next_conv_2,
    torch::Tensor* next_conv_1,
    torch::Tensor* next_scan) {
  PULSAR_TRACE_SCOPE_CAT("actor", "mamba2_block_step");
  torch::Tensor block_input = use_layer_norm_ ? input_norm_->forward(token) : token;
  const torch::Tensor weight = causal_conv_->weight.squeeze(1);
  torch::Tensor conv_out =
      previous_conv_2 * weight.select(1, 0).view({1, embed_dim_})
      + previous_conv_1 * weight.select(1, 1).view({1, embed_dim_})
      + block_input * weight.select(1, 2).view({1, embed_dim_});
  if (causal_conv_->bias.defined()) {
    conv_out = conv_out + causal_conv_->bias.view({1, embed_dim_});
  }
  conv_out = torch::silu(conv_out);

  const auto projected = input_projection_->forward(conv_out).chunk(5, -1);
  const torch::Tensor x = torch::silu(projected[0]);
  const torch::Tensor b = torch::sigmoid(projected[1]);
  const torch::Tensor c = torch::sigmoid(projected[2]);
  const torch::Tensor z = torch::silu(projected[3]);
  const torch::Tensor retention = torch::sigmoid(projected[4] + decay_bias_.view({1, embed_dim_}))
                                      .clamp(0.01, 0.9999);
  const torch::Tensor scan = retention * previous_scan + b * x;
  torch::Tensor mixed = (c * scan + skip_.view({1, embed_dim_}) * x) * z;
  if (use_layer_norm_) {
    mixed = output_norm_->forward(mixed);
  }

  if (next_conv_2 != nullptr) {
    *next_conv_2 = previous_conv_1;
  }
  if (next_conv_1 != nullptr) {
    *next_conv_1 = block_input;
  }
  if (next_scan != nullptr) {
    *next_scan = scan;
  }
  return token + output_projection_->forward(mixed);
}

Mamba2EncoderImpl::Mamba2EncoderImpl(const ModelConfig& config)
    : observation_dim_(config.observation_dim),
      embed_dim_(config.encoder_dim),
      sequence_length_(config.sequence_length > 0 ? config.sequence_length : config.observation_dim + 1) {
  input_projection_ = register_module("input_projection", torch::nn::Linear(observation_dim_, embed_dim_));

  blocks_.reserve(static_cast<std::size_t>(config.num_encoder_blocks));
  for (int i = 0; i < config.num_encoder_blocks; ++i) {
    Mamba2Block block(embed_dim_, sequence_length_, config.use_layer_norm);
    blocks_.push_back(register_module("block_" + std::to_string(i), block));
  }
  output_norm_ = register_module("output_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})));
}

torch::Tensor Mamba2EncoderImpl::forward(const torch::Tensor& obs) {
  PULSAR_TRACE_SCOPE_CAT("actor", "mamba2_encoder");
  torch::Tensor tokens = input_projection_->forward(obs).unsqueeze(1);
  for (Mamba2Block& block : blocks_) {
    tokens = block->forward(tokens);
  }
  tokens = output_norm_->forward(tokens);
  return tokens.squeeze(1);
}

torch::Tensor Mamba2EncoderImpl::forward_sequence(const torch::Tensor& obs_seq, const torch::Tensor& episode_starts) {
  PULSAR_TRACE_SCOPE_CAT("actor", "mamba2_encoder_sequence");
  const auto time = obs_seq.size(0);
  const auto batch = obs_seq.size(1);
  torch::Tensor tokens = input_projection_->forward(
      obs_seq.transpose(0, 1).reshape({batch * time, observation_dim_}))
      .reshape({batch, time, embed_dim_});
  torch::Tensor reset_mask = episode_starts.defined()
      ? episode_starts.transpose(0, 1).to(tokens.device()).to(tokens.scalar_type())
      : torch::Tensor{};
  for (Mamba2Block& block : blocks_) {
    tokens = block->forward(tokens, reset_mask);
  }
  tokens = output_norm_->forward(tokens);
  return tokens.transpose(0, 1);
}

torch::Tensor Mamba2EncoderImpl::initial_state(int64_t batch, const torch::Device& device) const {
  return torch::zeros(
      {static_cast<int64_t>(blocks_.size()), 3, batch, embed_dim_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

torch::Tensor Mamba2EncoderImpl::forward_step(
    const torch::Tensor& obs,
    const torch::Tensor& state,
    const torch::Tensor& episode_starts,
    torch::Tensor* next_state) {
  PULSAR_TRACE_SCOPE_CAT("actor", "mamba2_encoder_step");
  const auto batch = obs.size(0);
  torch::Tensor current_state = state.defined()
      ? state.to(obs.device()).to(torch::kFloat32)
      : initial_state(batch, obs.device());
  if (episode_starts.defined()) {
    const torch::Tensor keep = (1.0F - episode_starts.to(obs.device()).to(torch::kFloat32)).view({1, 1, batch, 1});
    current_state = current_state * keep;
  }

  torch::Tensor token = input_projection_->forward(obs);
  std::vector<torch::Tensor> next_blocks;
  next_blocks.reserve(blocks_.size());
  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    torch::Tensor next_conv_2;
    torch::Tensor next_conv_1;
    torch::Tensor next_scan;
    token = blocks_[i]->forward_step(
        token,
        current_state[static_cast<int64_t>(i)][0],
        current_state[static_cast<int64_t>(i)][1],
        current_state[static_cast<int64_t>(i)][2],
        &next_conv_2,
        &next_conv_1,
        &next_scan);
    next_blocks.push_back(torch::stack({next_conv_2, next_conv_1, next_scan}, 0));
  }
  if (next_state != nullptr) {
    *next_state = torch::stack(next_blocks, 0).detach();
  }
  return output_norm_->forward(token);
}

GoalCriticImpl::GoalCriticImpl(int feature_dim, int action_dim, int embedding_dim, int hidden_dim, int goal_dim)
    : action_dim_(action_dim), hidden_dim_(hidden_dim), embedding_dim_(embedding_dim), goal_dim_(goal_dim) {
  sa_encoder_ = torch::nn::Sequential();
  sa_encoder_->push_back(torch::nn::Linear(feature_dim + action_dim, hidden_dim));
  sa_encoder_->push_back(torch::nn::Functional(torch::relu));
  sa_encoder_->push_back(torch::nn::Linear(hidden_dim, embedding_dim));
  register_module("sa_encoder", sa_encoder_);

  goal_encoder_ = torch::nn::Sequential();
  goal_encoder_->push_back(torch::nn::Linear(goal_dim, hidden_dim));
  goal_encoder_->push_back(torch::nn::Functional(torch::relu));
  goal_encoder_->push_back(torch::nn::Linear(hidden_dim, embedding_dim));
  register_module("goal_encoder", goal_encoder_);
}

torch::Tensor GoalCriticImpl::forward(
    const torch::Tensor& features,
    const torch::Tensor& action_inputs,
    const torch::Tensor& goal_value) {
  PULSAR_TRACE_SCOPE_CAT("actor", "goal_forward");
  return -((sa_embedding(features, action_inputs) - goal_embedding(goal_value)).square().sum(-1).clamp_min(1.0e-8F));
}

torch::Tensor GoalCriticImpl::sa_embedding(const torch::Tensor& features, const torch::Tensor& action_inputs) {
  PULSAR_TRACE_SCOPE_CAT("actor", "sa_embedding");
  const torch::Tensor action_tensor = action_inputs.dim() == 1
      ? torch::nn::functional::one_hot(action_inputs.to(torch::kLong), action_dim_).to(features.device()).to(torch::kFloat32)
      : action_inputs.to(features.device()).to(torch::kFloat32);
  return sa_encoder_->forward(torch::cat({features, action_tensor}, -1));
}

torch::Tensor GoalCriticImpl::goal_embedding(const torch::Tensor& goal_values) {
  PULSAR_TRACE_SCOPE_CAT("actor", "goal_embedding");
  return goal_encoder_->forward(goal_values.to(torch::kFloat32));
}

torch::nn::Sequential PPOActorImpl::make_value_win_head(int input_dim) const {
  torch::nn::Sequential head = torch::nn::Sequential();
  head->push_back(torch::nn::Linear(input_dim, config_.value_hidden_dim));
  head->push_back(torch::nn::Functional(torch::relu));
  head->push_back(torch::nn::Linear(config_.value_hidden_dim, 1));
  return head;
}

PPOActorImpl::PPOActorImpl(
    ModelConfig config,
    const GoalCriticConfig& goal_critic_config,
    const ESLoraConfig& es_lora_config)
    : config_(std::move(config)),
      goal_critic_config_(goal_critic_config),
      es_lora_config_(es_lora_config) {
  validate_model_config(config_);
  validate_es_lora_config(es_lora_config_);

  mamba2_encoder_ = Mamba2Encoder(config_);
  register_module("encoder", mamba2_encoder_);

  feature_dim_ = config_.encoder_dim;

  if (config_.policy_hidden_dim > 0) {
    policy_hidden_ = torch::nn::Sequential();
    policy_hidden_->push_back(torch::nn::Linear(feature_dim_, config_.policy_hidden_dim));
    policy_hidden_->push_back(torch::nn::Functional(torch::relu));
    register_module("policy_hidden", policy_hidden_);
    policy_lora_ = LoRALinear(
        config_.policy_hidden_dim, config_.action_dim, es_lora_config_.rank, es_lora_config_.lora_alpha);
  } else {
    policy_lora_ = LoRALinear(feature_dim_, config_.action_dim, es_lora_config_.rank, es_lora_config_.lora_alpha);
  }
  register_module("policy_lora", policy_lora_);

  value_head_win_ = make_value_win_head(feature_dim_);
  register_module("value_head_win", value_head_win_);

  goal_critic_ = GoalCritic(feature_dim_, config_.action_dim, goal_critic_config_.embedding_dim, goal_critic_config_.hidden_dim, goal_critic_config_.goal_dim);
  register_module("goal_critic", goal_critic_);
}

ActorStepOutput PPOActorImpl::forward_step(
    torch::Tensor obs,
    torch::Tensor goal_values) {
  PULSAR_TRACE_SCOPE_CAT("actor", "forward_step");
  torch::Tensor encoded = mamba2_encoder_->forward(obs);

  torch::Tensor policy_logits;
  if (!policy_hidden_.is_empty()) {
    policy_logits = policy_lora_->forward(policy_hidden_->forward(encoded));
  } else {
    policy_logits = policy_lora_->forward(encoded);
  }

  return {
      policy_logits,
      encoded,
      value_head_win_->forward(encoded),
      encoded,
  };
}

ActorStepOutput PPOActorImpl::forward_step_stateful(
    torch::Tensor obs,
    torch::Tensor state,
    torch::Tensor episode_starts,
    torch::Tensor* next_state,
    torch::Tensor goal_values) {
  PULSAR_TRACE_SCOPE_CAT("actor", "forward_step_stateful");
  torch::Tensor encoded = mamba2_encoder_->forward_step(obs, state, episode_starts, next_state);

  torch::Tensor policy_logits;
  if (!policy_hidden_.is_empty()) {
    policy_logits = policy_lora_->forward(policy_hidden_->forward(encoded));
  } else {
    policy_logits = policy_lora_->forward(encoded);
  }

  return {
      policy_logits,
      encoded,
      value_head_win_->forward(encoded),
      encoded,
  };
}

ActorSequenceOutput PPOActorImpl::forward_sequence(
    torch::Tensor obs_seq,
    torch::Tensor goal_values,
    torch::Tensor episode_starts) {
  PULSAR_TRACE_SCOPE_CAT("actor", "forward_sequence");
  const auto time = obs_seq.size(0);
  const auto batch = obs_seq.size(1);
  torch::Tensor encoded = mamba2_encoder_->forward_sequence(obs_seq, episode_starts).reshape({time * batch, config_.encoder_dim});

  torch::Tensor policy_logits;
  if (!policy_hidden_.is_empty()) {
    policy_logits = policy_lora_->forward(policy_hidden_->forward(encoded));
  } else {
    policy_logits = policy_lora_->forward(encoded);
  }

  return {
      policy_logits.reshape({time, batch, config_.action_dim}),
      encoded.reshape({time, batch, config_.encoder_dim}),
      value_head_win_->forward(encoded).reshape({time, batch, 1}),
      encoded.reshape({time, batch, config_.encoder_dim}),
  };
}

torch::Tensor PPOActorImpl::initial_recurrent_state(int64_t batch, const torch::Device& device) const {
  return mamba2_encoder_->initial_state(batch, device);
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

const ESLoraConfig& PPOActorImpl::es_lora_config() const {
  return es_lora_config_;
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
    float sigma,
    torch::Tensor /* goal_values */) {
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

  torch::Device torch_device(device);
  auto model = PPOActor(config.model, config.goal_critic, config.es_lora);
  const fs::path state_path = base / "state.pt";
  if (fs::exists(state_path)) {
    torch::serialize::InputArchive archive;
    archive.load_from(state_path.string(), torch_device);
    model->load(archive);
  } else {
    torch::serialize::InputArchive archive;
    archive.load_from((base / "model.pt").string(), torch_device);
    model->load(archive);
  }
  model->to(torch_device);
  model->eval();
  return model;
}

PPOActor clone_ppo_actor(const PPOActor& source, const torch::Device& device) {
  if (!source) {
    return nullptr;
  }
  auto clone = PPOActor(source->config(), source->goal_critic_config(), source->es_lora_config());
  clone->to(device);
  copy_module_tensors_to(source, clone, device);
  return clone;
}

void copy_ppo_actor_tensors_to(const PPOActor& source, PPOActor& target, const torch::Device& device) {
  if (!source || !target) {
    throw std::invalid_argument("copy_ppo_actor_tensors_to requires non-null actors.");
  }
  copy_module_tensors_to(source, target, device);
}

}  // namespace pulsar

#endif
