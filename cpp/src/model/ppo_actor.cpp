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

void copy_module_tensors_to(const PPOActor& source, PPOActor& target, const torch::Device& device) {
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
  require_positive(config.transformer_num_heads, "transformer_num_heads");
  require_positive(config.transformer_window_size, "transformer_window_size");
  require_positive(config.transformer_max_batch_size, "transformer_max_batch_size");
  require_positive(config.transformer_token_group_size, "transformer_token_group_size");
  require_positive(config.transformer_ffn_multiplier, "transformer_ffn_multiplier");
  require_positive(config.value_hidden_dim, "value_hidden_dim");
  if (config.encoder_dim % config.transformer_num_heads != 0) {
    throw std::invalid_argument("ModelConfig.encoder_dim must be divisible by transformer_num_heads.");
  }
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

SlidingWindowSelfAttentionImpl::SlidingWindowSelfAttentionImpl(
    int embed_dim,
    int num_heads,
    int window_size,
    int sequence_length)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      window_size_(window_size),
      sequence_length_(sequence_length) {
  if (embed_dim_ <= 0 || num_heads_ <= 0 || head_dim_ * num_heads_ != embed_dim_) {
    throw std::invalid_argument("SlidingWindowSelfAttention requires embed_dim divisible by num_heads.");
  }
  if (window_size_ <= 0 || sequence_length_ <= 0) {
    throw std::invalid_argument("SlidingWindowSelfAttention requires positive window and sequence length.");
  }

  qkv_ = register_module("qkv", torch::nn::Linear(embed_dim_, 3 * embed_dim_));
  out_proj_ = register_module("out_proj", torch::nn::Linear(embed_dim_, embed_dim_));

  torch::Tensor mask = torch::zeros(
      {sequence_length_, sequence_length_},
      torch::TensorOptions().dtype(torch::kBool));
  auto mask_accessor = mask.accessor<bool, 2>();
  for (int query = 0; query < sequence_length_; ++query) {
    for (int key = 0; key < sequence_length_; ++key) {
      const bool global_cls = query == 0 || key == 0;
      const bool local_window = std::abs(query - key) <= window_size_;
      if (global_cls || local_window) {
        mask_accessor[query][key] = true;
      }
    }
  }
  attention_mask_ = register_buffer("attention_mask", mask.view({1, 1, sequence_length_, sequence_length_}));
}

torch::Tensor SlidingWindowSelfAttentionImpl::forward(const torch::Tensor& tokens) {
  PULSAR_TRACE_SCOPE_CAT("actor", "swa_attention");
  const auto batch = tokens.size(0);
  const auto sequence = tokens.size(1);

  torch::Tensor qkv = qkv_->forward(tokens)
      .view({batch, sequence, 3, num_heads_, head_dim_})
      .permute({2, 0, 3, 1, 4});
  const auto parts = qkv.unbind(0);
  const torch::Tensor q = parts[0];
  const torch::Tensor k = parts[1];
  const torch::Tensor v = parts[2];

  const torch::Tensor mask = attention_mask_.to(q.device()).narrow(2, 0, sequence).narrow(3, 0, sequence);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
  torch::Tensor attention_scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
  attention_scores = attention_scores.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity());
  torch::Tensor attended = torch::matmul(torch::softmax(attention_scores, -1), v)
      .transpose(1, 2)
      .contiguous()
      .view({batch, sequence, embed_dim_});
  return out_proj_->forward(attended);
}

SWATransformerBlockImpl::SWATransformerBlockImpl(
    int embed_dim,
    int num_heads,
    int window_size,
    int sequence_length,
    int ffn_multiplier,
    bool use_layer_norm)
    : use_layer_norm_(use_layer_norm) {
  attention_ = register_module(
      "attention",
      SlidingWindowSelfAttention(embed_dim, num_heads, window_size, sequence_length));
  if (use_layer_norm_) {
    attn_norm_ = register_module("attn_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    ffn_norm_ = register_module("ffn_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
  }
  ffn_ = torch::nn::Sequential();
  const int ffn_dim = ffn_multiplier * embed_dim;
  ffn_->push_back(torch::nn::Linear(embed_dim, ffn_dim));
  ffn_->push_back(torch::nn::Functional(torch::relu));
  ffn_->push_back(torch::nn::Linear(ffn_dim, embed_dim));
  register_module("ffn", ffn_);
}

torch::Tensor SWATransformerBlockImpl::forward(const torch::Tensor& tokens) {
  PULSAR_TRACE_SCOPE_CAT("actor", "swa_block");
  torch::Tensor attn_input = use_layer_norm_ ? attn_norm_->forward(tokens) : tokens;
  torch::Tensor hidden = tokens + attention_->forward(attn_input);
  torch::Tensor ffn_input = use_layer_norm_ ? ffn_norm_->forward(hidden) : hidden;
  return hidden + ffn_->forward(ffn_input);
}

SWATransformerEncoderImpl::SWATransformerEncoderImpl(const ModelConfig& config)
    : observation_dim_(config.observation_dim),
      token_group_size_(config.transformer_token_group_size),
      padded_observation_dim_(((config.observation_dim + config.transformer_token_group_size - 1) /
                               config.transformer_token_group_size) *
                              config.transformer_token_group_size),
      embed_dim_(config.encoder_dim),
      sequence_length_(padded_observation_dim_ / token_group_size_ + 1) {
  input_projection_ = register_module("input_projection", torch::nn::Linear(token_group_size_, embed_dim_));
  cls_token_ = register_parameter("cls_token", torch::zeros({1, 1, embed_dim_}));
  position_embedding_ = register_parameter(
      "position_embedding",
      torch::randn({1, sequence_length_, embed_dim_}) * 0.01F);

  blocks_.reserve(static_cast<std::size_t>(config.num_encoder_blocks));
  for (int i = 0; i < config.num_encoder_blocks; ++i) {
    SWATransformerBlock block(
        embed_dim_,
        config.transformer_num_heads,
        config.transformer_window_size,
        sequence_length_,
        config.transformer_ffn_multiplier,
        config.use_layer_norm);
    blocks_.push_back(register_module("block_" + std::to_string(i), block));
  }
  output_norm_ = register_module("output_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})));
}

torch::Tensor SWATransformerEncoderImpl::forward(const torch::Tensor& obs) {
  PULSAR_TRACE_SCOPE_CAT("actor", "swa_encoder");
  const auto batch = obs.size(0);
  torch::Tensor grouped_obs = obs;
  if (padded_observation_dim_ != observation_dim_) {
    grouped_obs = torch::cat(
        {obs, torch::zeros({batch, padded_observation_dim_ - observation_dim_}, obs.options())},
        1);
  }
  torch::Tensor tokens = input_projection_->forward(
      grouped_obs.view({batch, padded_observation_dim_ / token_group_size_, token_group_size_}));
  const torch::Tensor cls = cls_token_.expand({batch, -1, -1});
  tokens = torch::cat({cls, tokens}, 1) + position_embedding_.to(obs.device());
  for (SWATransformerBlock& block : blocks_) {
    tokens = block->forward(tokens);
  }
  tokens = output_norm_->forward(tokens);
  return tokens.select(1, 0);
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

  encoder_ = SWATransformerEncoder(config_);
  register_module("encoder", encoder_);

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
  torch::Tensor encoded = encoder_->forward(obs);

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
    torch::Tensor goal_values) {
  PULSAR_TRACE_SCOPE_CAT("actor", "forward_sequence");
  const auto time = obs_seq.size(0);
  const auto batch = obs_seq.size(1);
  const auto flat_batch = time * batch;
  torch::Tensor encoded = encoder_->forward(obs_seq.reshape({flat_batch, config_.observation_dim}));

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

}  // namespace pulsar

#endif
