#include "pulsar/model/vrpo_actor.hpp"

#include "pulsar/model/layernorm_ops.hpp"

#ifdef PULSAR_HAS_TORCH

#include <cstdint>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/model/mamba2_ops.hpp"
#include "pulsar/tracing/tracing.hpp"

namespace pulsar {

#ifdef PULSAR_HAS_LORA_CUDA_KERNELS
torch::Tensor eggroll_lora_perturb_cuda(
    const torch::Tensor& base_out,
    const torch::Tensor& x_A_stack,
    const torch::Tensor& B_stack,
    float perturb_scale);
#endif

#ifdef PULSAR_HAS_LORA_HIP_KERNELS
torch::Tensor eggroll_lora_perturb_hip(
    const torch::Tensor& base_out,
    const torch::Tensor& x_A_stack,
    const torch::Tensor& B_stack,
    float perturb_scale);
#endif

namespace {

void copy_module_tensors_to(const VRPOActor& source, VRPOActor& target, const torch::Device&) {
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

torch::Tensor normalize_goal_embedding(const torch::Tensor& embedding) {
  constexpr float kNormEps = 0.1F;
  const torch::Tensor denom = (embedding.square().sum(-1, true) + kNormEps * kNormEps).sqrt();
  return embedding / denom;
}

bool can_use_eggroll_lora_accel(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& A_stack,
    const torch::Tensor& B_stack,
    int64_t population,
    int64_t rank) {
#if defined(PULSAR_HAS_LORA_CUDA_KERNELS) || defined(PULSAR_HAS_LORA_HIP_KERNELS)
  return x.defined() &&
      x.is_cuda() &&
      (x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16) &&
      weight.is_cuda() &&
      weight.scalar_type() == x.scalar_type() &&
      A.is_cuda() &&
      B.is_cuda() &&
      A_stack.is_cuda() &&
      B_stack.is_cuda() &&
      A.scalar_type() == x.scalar_type() &&
      B.scalar_type() == x.scalar_type() &&
      A_stack.scalar_type() == x.scalar_type() &&
      B_stack.scalar_type() == x.scalar_type() &&
      x.dim() == 2 &&
      weight.dim() == 2 &&
      A.dim() == 2 &&
      B.dim() == 2 &&
      A_stack.dim() == 3 &&
      B_stack.dim() == 3 &&
      A.size(0) == rank &&
      B.size(1) == rank &&
      A_stack.size(0) == population &&
      A_stack.size(1) == rank &&
      A_stack.size(2) == x.size(1) &&
      B_stack.size(0) == population &&
      B_stack.size(1) == weight.size(0) &&
      B_stack.size(2) == rank &&
      B.size(0) == weight.size(0) &&
      x.size(1) == weight.size(1);
#else
  (void)x;
  (void)weight;
  (void)A;
  (void)B;
  (void)A_stack;
  (void)B_stack;
  (void)population;
  (void)rank;
  return false;
#endif
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
  if (can_use_eggroll_lora_accel(x, base->weight, A, B, A_stack, B_stack, population, rank_)) {
    torch::Tensor x_contig = x.contiguous();
    torch::Tensor base_out = forward(x_contig);
    torch::Tensor x_A_stack = torch::bmm(
        x_contig.view({population, member_batch, in_features()}),
        A_stack.transpose(1, 2)).view({x.size(0), rank_});
    const float perturb_scale = sigma / std::sqrt(static_cast<float>(rank_));
#ifdef PULSAR_HAS_LORA_CUDA_KERNELS
    return eggroll_lora_perturb_cuda(
        base_out.contiguous(),
        x_A_stack.contiguous(),
        B_stack.contiguous(),
        perturb_scale);
#elif defined(PULSAR_HAS_LORA_HIP_KERNELS)
    return eggroll_lora_perturb_hip(
        base_out.contiguous(),
        x_A_stack.contiguous(),
        B_stack.contiguous(),
        perturb_scale);
#endif
  }
  torch::Tensor base_out = forward(x).view({population, member_batch, out_features()});
  torch::Tensor x_view = x.view({population, member_batch, in_features()});

  torch::Tensor x_A_stack = torch::bmm(x_view, A_stack.transpose(1, 2));
  torch::Tensor perturbation = torch::bmm(x_A_stack, B_stack.transpose(1, 2));
  const float perturb_scale = sigma / std::sqrt(static_cast<float>(rank_));
  return (base_out + perturb_scale * perturbation)
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
  decay_bias_ = register_parameter("decay_bias", torch::full({embed_dim_}, 4.0F));
  skip_ = register_parameter("skip", torch::full({embed_dim_}, 0.1F));
}

torch::Tensor Mamba2BlockImpl::forward(const torch::Tensor& tokens, const torch::Tensor& reset_mask) {
  PULSAR_TRACE_SCOPE_CAT("actor", "mamba2_block");
  const auto batch = tokens.size(0);
  const auto sequence = tokens.size(1);
  torch::Tensor block_input = use_layer_norm_ ? fused_layer_norm(tokens, input_norm_) : tokens;

  const torch::Tensor reset = reset_mask.defined()
      ? reset_mask.to(tokens.device()).to(tokens.scalar_type())
      : torch::Tensor{};

  torch::Tensor conv_out;
  if (reset.defined()) {
    const torch::Tensor weight = causal_conv_->weight.squeeze(1);
    conv_out = mamba2_causal_conv1d_silu(block_input, weight, causal_conv_->bias, reset);
  } else {
    torch::Tensor conv_input = block_input.transpose(1, 2);
    conv_out = causal_conv_->forward(conv_input).narrow(2, 0, sequence).transpose(1, 2);
    conv_out = torch::silu(conv_out);
  }

  const torch::Tensor projected = input_projection_->forward(conv_out);
  torch::Tensor mixed = mamba2_scan_mixed(projected, decay_bias_, skip_, reset);
  if (use_layer_norm_) {
    mixed = fused_layer_norm(mixed, output_norm_);
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
  torch::Tensor block_input = use_layer_norm_ ? fused_layer_norm(token, input_norm_) : token;
  const torch::Tensor weight = causal_conv_->weight.squeeze(1);
  torch::Tensor conv_out =
      previous_conv_2 * weight.select(1, 0).view({1, embed_dim_})
      + previous_conv_1 * weight.select(1, 1).view({1, embed_dim_})
      + block_input * weight.select(1, 2).view({1, embed_dim_});
  if (causal_conv_->bias.defined()) {
    conv_out = conv_out + causal_conv_->bias.view({1, embed_dim_});
  }
  conv_out = torch::silu(conv_out);

  const torch::Tensor projected = input_projection_->forward(conv_out);
  auto [mixed, scan] = mamba2_step_mixed(projected, previous_scan, decay_bias_, skip_);
  if (use_layer_norm_) {
    mixed = fused_layer_norm(mixed, output_norm_);
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
    tokens = torch::clamp(tokens, -100.0, 100.0);
  }
  tokens = fused_layer_norm(tokens, output_norm_);
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
    tokens = torch::clamp(tokens, -100.0, 100.0);
  }
  tokens = fused_layer_norm(tokens, output_norm_);
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
  torch::Tensor next_state_tensor;
  if (next_state != nullptr) {
    if (next_state->defined() &&
        next_state->sizes() == current_state.sizes() &&
        next_state->device() == current_state.device() &&
        next_state->scalar_type() == current_state.scalar_type()) {
      next_state_tensor = *next_state;
    } else {
      next_state_tensor = torch::empty_like(current_state);
    }
  }
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
    if (next_state_tensor.defined()) {
      torch::NoGradGuard no_grad;
      next_state_tensor[static_cast<int64_t>(i)][0].copy_(next_conv_2);
      next_state_tensor[static_cast<int64_t>(i)][1].copy_(next_conv_1);
      next_state_tensor[static_cast<int64_t>(i)][2].copy_(next_scan);
    }
  }
  if (next_state != nullptr) {
    *next_state = next_state_tensor.detach();
  }
  return fused_layer_norm(token, output_norm_);
}

GoalCriticImpl::GoalCriticImpl(int feature_dim, int action_dim, int embedding_dim, int hidden_dim, int goal_dim)
    : action_dim_(action_dim), hidden_dim_(hidden_dim), embedding_dim_(embedding_dim), goal_dim_(goal_dim) {
  sa_encoder_ = torch::nn::Sequential();
  sa_encoder_->push_back(torch::nn::Linear(feature_dim + action_dim, hidden_dim));
  sa_encoder_->push_back(torch::nn::Functional(torch::silu));
  sa_encoder_->push_back(torch::nn::Linear(hidden_dim, embedding_dim));
  register_module("sa_encoder", sa_encoder_);

  goal_encoder_ = torch::nn::Sequential();
  goal_encoder_->push_back(torch::nn::Linear(goal_dim, hidden_dim));
  goal_encoder_->push_back(torch::nn::Functional(torch::silu));
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
  return normalize_goal_embedding(sa_encoder_->forward(torch::cat({features, action_tensor}, -1)));
}

torch::Tensor GoalCriticImpl::goal_embedding(const torch::Tensor& goal_values) {
  PULSAR_TRACE_SCOPE_CAT("actor", "goal_embedding");
  return normalize_goal_embedding(goal_encoder_->forward(goal_values.to(torch::kFloat32)));
}

VRPOActorImpl::VRPOActorImpl(
    ModelConfig config,
    const GoalCriticConfig& goal_critic_config,
    const ESLoraConfig& es_lora_config,
    const VRPOConfig& vrpo_config)
    : config_(std::move(config)),
      goal_critic_config_(goal_critic_config),
      es_lora_config_(es_lora_config),
      vrpo_config_(vrpo_config) {
  validate_model_config(config_);
  validate_es_lora_config(es_lora_config_);

  mamba2_encoder_ = Mamba2Encoder(config_);
  register_module("encoder", mamba2_encoder_);

  feature_dim_ = config_.encoder_dim;

  if (config_.policy_hidden_dim > 0) {
    policy_hidden_ = torch::nn::Sequential();
    policy_hidden_->push_back(torch::nn::Linear(feature_dim_, config_.policy_hidden_dim));
    policy_hidden_->push_back(torch::nn::Functional(torch::silu));
    register_module("policy_hidden", policy_hidden_);
    policy_lora_ = LoRALinear(
        config_.policy_hidden_dim, config_.action_dim, es_lora_config_.rank, es_lora_config_.lora_alpha);
  } else {
    policy_lora_ = LoRALinear(feature_dim_, config_.action_dim, es_lora_config_.rank, es_lora_config_.lora_alpha);
  }
  register_module("policy_lora", policy_lora_);

  q_head_ = register_module("q_head",
      torch::nn::Sequential(
          torch::nn::Linear(feature_dim_, vrpo_config_.q_critic_hidden_dim),
          torch::nn::SiLU(),
          torch::nn::Linear(vrpo_config_.q_critic_hidden_dim, config_.action_dim)
      ));

  q_head_sparse_ = register_module("q_head_sparse",
      torch::nn::Sequential(
          torch::nn::Linear(feature_dim_, vrpo_config_.q_critic_hidden_dim),
          torch::nn::SiLU(),
          torch::nn::Linear(vrpo_config_.q_critic_hidden_dim, config_.action_dim)
      ));

  goal_critic_ = GoalCritic(feature_dim_, config_.action_dim, goal_critic_config_.embedding_dim, goal_critic_config_.hidden_dim, goal_critic_config_.goal_dim);
  register_module("goal_critic", goal_critic_);

  predictor_heads_.clear();
  predictor_heads_.reserve(kSparseEventChannelCount);
  predictor_active_.assign(kSparseEventChannelCount, 0);
  for (const auto& channel : kSparseEventChannels) {
    SparseRewardPredictor head(feature_dim_);
    register_module("predictor_" + std::string(channel.name), head);
    predictor_heads_.push_back(head);
  }
}

SparseRewardPredictor& VRPOActorImpl::predictor_head(std::size_t channel_index) {
  if (channel_index >= predictor_heads_.size()) {
    throw std::out_of_range("predictor channel index is out of range");
  }
  return predictor_heads_[channel_index];
}

const SparseRewardPredictor& VRPOActorImpl::predictor_head(std::size_t channel_index) const {
  if (channel_index >= predictor_heads_.size()) {
    throw std::out_of_range("predictor channel index is out of range");
  }
  return predictor_heads_[channel_index];
}

torch::Tensor VRPOActorImpl::forward_all_predictors(const torch::Tensor& features) const {
  if (predictor_heads_.empty()) {
    return torch::empty({features.size(0), 0, 2}, features.options());
  }

  std::vector<torch::Tensor> input_norm_weights;
  std::vector<torch::Tensor> input_norm_biases;
  std::vector<torch::Tensor> input_linear_weights;
  std::vector<torch::Tensor> input_linear_biases;
  std::vector<torch::Tensor> hidden_norm_weights;
  std::vector<torch::Tensor> hidden_norm_biases;
  std::vector<torch::Tensor> output_linear_weights;
  std::vector<torch::Tensor> output_linear_biases;
  input_norm_weights.reserve(predictor_heads_.size());
  input_norm_biases.reserve(predictor_heads_.size());
  input_linear_weights.reserve(predictor_heads_.size());
  input_linear_biases.reserve(predictor_heads_.size());
  hidden_norm_weights.reserve(predictor_heads_.size());
  hidden_norm_biases.reserve(predictor_heads_.size());
  output_linear_weights.reserve(predictor_heads_.size());
  output_linear_biases.reserve(predictor_heads_.size());

  for (const auto& head : predictor_heads_) {
    const auto& input_norm = head->input_norm();
    const auto& input_linear = head->input_linear();
    const auto& hidden_norm = head->hidden_norm();
    const auto& output_linear = head->output_linear();
    input_norm_weights.push_back(input_norm.weight);
    input_norm_biases.push_back(input_norm.bias);
    input_linear_weights.push_back(input_linear.weight);
    input_linear_biases.push_back(input_linear.bias);
    hidden_norm_weights.push_back(hidden_norm.weight);
    hidden_norm_biases.push_back(hidden_norm.bias);
    output_linear_weights.push_back(output_linear.weight);
    output_linear_biases.push_back(output_linear.bias);
  }

  constexpr double kLayerNormEps = 1.0e-5;
  auto normalize_last_dim = [](const torch::Tensor& input) {
    const torch::Tensor mean = input.mean(-1, true);
    const torch::Tensor centered = input - mean;
    const torch::Tensor variance = centered.square().mean(-1, true);
    return centered * torch::rsqrt(variance + kLayerNormEps);
  };

  const torch::Tensor input_normalized = normalize_last_dim(features);
  torch::Tensor hidden = input_normalized.unsqueeze(0) * torch::stack(input_norm_weights).unsqueeze(1) +
      torch::stack(input_norm_biases).unsqueeze(1);
  hidden = torch::bmm(hidden, torch::stack(input_linear_weights).transpose(1, 2)) +
      torch::stack(input_linear_biases).unsqueeze(1);
  hidden = torch::relu(hidden);
  hidden = normalize_last_dim(hidden);
  hidden = hidden * torch::stack(hidden_norm_weights).unsqueeze(1) +
      torch::stack(hidden_norm_biases).unsqueeze(1);
  torch::Tensor logits = torch::bmm(hidden, torch::stack(output_linear_weights).transpose(1, 2)) +
      torch::stack(output_linear_biases).unsqueeze(1);
  return torch::clamp(logits.permute({1, 0, 2}), -10.0, 10.0);
}

ActorStepOutput VRPOActorImpl::forward_step(
    torch::Tensor obs,
    torch::Tensor goal_values,
    bool compute_value) {
  PULSAR_TRACE_SCOPE_CAT("actor", "forward_step");
  torch::Tensor encoded = mamba2_encoder_->forward(obs);

  torch::Tensor policy_logits;
  if (!policy_hidden_.is_empty()) {
    policy_logits = policy_lora_->forward(policy_hidden_->forward(encoded));
  } else {
    policy_logits = policy_lora_->forward(encoded);
  }

  torch::Tensor q_values = compute_value ? q_head_forward(encoded) : torch::Tensor{};
  torch::Tensor expected_value;
  torch::Tensor q_values_sparse = compute_value ? q_head_sparse_forward(encoded) : torch::Tensor{};
  torch::Tensor expected_value_sparse;
  if (compute_value) {
    torch::Tensor policy_probs = torch::softmax(policy_logits, -1);
    expected_value = (policy_probs * q_values).sum(-1, true);
    expected_value_sparse = (policy_probs * q_values_sparse).sum(-1, true);
  }

  return {
      policy_logits,
      encoded,
      expected_value,
      encoded,
      q_values,
      expected_value_sparse,
      q_values_sparse,
  };
}

ActorStepOutput VRPOActorImpl::forward_step_stateful(
    torch::Tensor obs,
    torch::Tensor state,
    torch::Tensor episode_starts,
    torch::Tensor* next_state,
    torch::Tensor goal_values,
    bool compute_value) {
  PULSAR_TRACE_SCOPE_CAT("actor", "forward_step_stateful");
  torch::Tensor encoded = mamba2_encoder_->forward_step(obs, state, episode_starts, next_state);

  torch::Tensor policy_logits;
  if (!policy_hidden_.is_empty()) {
    policy_logits = policy_lora_->forward(policy_hidden_->forward(encoded));
  } else {
    policy_logits = policy_lora_->forward(encoded);
  }

  torch::Tensor q_values = compute_value ? q_head_forward(encoded) : torch::Tensor{};
  torch::Tensor expected_value;
  torch::Tensor q_values_sparse = compute_value ? q_head_sparse_forward(encoded) : torch::Tensor{};
  torch::Tensor expected_value_sparse;
  if (compute_value) {
    torch::Tensor policy_probs = torch::softmax(policy_logits, -1);
    expected_value = (policy_probs * q_values).sum(-1, true);
    expected_value_sparse = (policy_probs * q_values_sparse).sum(-1, true);
  }

  return {
      policy_logits,
      encoded,
      expected_value,
      encoded,
      q_values,
      expected_value_sparse,
      q_values_sparse,
  };
}

torch::Tensor VRPOActorImpl::value_head_forward(const torch::Tensor& encoded) {
  auto policy_logits = policy_head_forward(encoded);
  auto q_values = q_head_forward(encoded);
  auto policy_probs = torch::softmax(policy_logits, -1);
  return (policy_probs * q_values).sum(-1, true);
}

torch::Tensor VRPOActorImpl::q_head_forward(const torch::Tensor& encoded) {
  const float q_value_abs_cap = std::max(vrpo_config_.q_value_abs_cap, 1.0e-6F);
  const torch::Tensor raw_q = q_head_->forward(encoded);
  return torch::tanh(raw_q / q_value_abs_cap) * q_value_abs_cap;
}

torch::Tensor VRPOActorImpl::value_head_sparse_forward(const torch::Tensor& encoded) {
  auto policy_logits = policy_head_forward(encoded);
  auto q_values = q_head_sparse_forward(encoded);
  auto policy_probs = torch::softmax(policy_logits, -1);
  return (policy_probs * q_values).sum(-1, true);
}

torch::Tensor VRPOActorImpl::q_head_sparse_forward(const torch::Tensor& encoded) {
  const float q_value_abs_cap = std::max(vrpo_config_.q_value_abs_cap, 1.0e-6F);
  const torch::Tensor raw_q = q_head_sparse_->forward(encoded);
  return torch::tanh(raw_q / q_value_abs_cap) * q_value_abs_cap;
}

torch::Tensor VRPOActorImpl::policy_head_forward(const torch::Tensor& features) {
  if (!policy_hidden_.is_empty()) {
    return policy_lora_->forward(policy_hidden_->forward(features));
  }
  return policy_lora_->forward(features);
}

ActorSequenceOutput VRPOActorImpl::forward_sequence(
    torch::Tensor obs_seq,
    torch::Tensor goal_values,
    torch::Tensor episode_starts,
    bool compute_value) {
  PULSAR_TRACE_SCOPE_CAT("actor", "forward_sequence");
  const auto time = obs_seq.size(0);
  const auto batch = obs_seq.size(1);
  torch::Tensor encoded_seq = mamba2_encoder_->forward_sequence(obs_seq, episode_starts);
  torch::Tensor encoded = encoded_seq.reshape({time * batch, config_.encoder_dim});

  torch::Tensor policy_logits;
  if (!policy_hidden_.is_empty()) {
    policy_logits = policy_lora_->forward(policy_hidden_->forward(encoded));
  } else {
    policy_logits = policy_lora_->forward(encoded);
  }

  torch::Tensor q_values = compute_value ? q_head_forward(encoded) : torch::Tensor{};
  torch::Tensor expected_value;
  torch::Tensor q_values_sparse = compute_value ? q_head_sparse_forward(encoded) : torch::Tensor{};
  torch::Tensor expected_value_sparse;
  if (compute_value) {
    torch::Tensor policy_probs = torch::softmax(policy_logits, -1);
    expected_value = (policy_probs * q_values).sum(-1, true).reshape({time, batch, 1});
    expected_value_sparse = (policy_probs * q_values_sparse).sum(-1, true).reshape({time, batch, 1});
  }

  return {
      policy_logits.reshape({time, batch, config_.action_dim}),
      encoded_seq,
      expected_value,
      encoded_seq,
      compute_value ? q_values.reshape({time, batch, config_.action_dim}) : torch::Tensor{},
      expected_value_sparse,
      compute_value ? q_values_sparse.reshape({time, batch, config_.action_dim}) : torch::Tensor{},
  };
}

torch::Tensor VRPOActorImpl::initial_recurrent_state(int64_t batch, const torch::Device& device) const {
  return mamba2_encoder_->initial_state(batch, device);
}

int VRPOActorImpl::feature_dim() const {
  return feature_dim_;
}

const ModelConfig& VRPOActorImpl::config() const {
  return config_;
}

const GoalCriticConfig& VRPOActorImpl::goal_critic_config() const {
  return goal_critic_config_;
}

const ESLoraConfig& VRPOActorImpl::es_lora_config() const {
  return es_lora_config_;
}

const VRPOConfig& VRPOActorImpl::vrpo_config() const {
  return vrpo_config_;
}

std::vector<torch::Tensor> VRPOActorImpl::es_lora_parameters() const {
  return policy_lora_->lora_parameters();
}

std::vector<torch::Tensor> VRPOActorImpl::es_lora_parameters_flat() const {
  return policy_lora_->lora_parameters_flat();
}

void VRPOActorImpl::restore_es_lora_parameters(const std::vector<torch::Tensor>& params) {
  policy_lora_->restore_lora_parameters(params);
}

void VRPOActorImpl::apply_lora_perturbation(
    const std::vector<torch::Tensor>& perturbation, float sigma) {
  torch::NoGradGuard no_grad;
  auto params = es_lora_parameters();
  for (std::size_t i = 0; i < params.size(); ++i) {
    params[i].add_(perturbation[i], sigma);
  }
}

torch::Tensor VRPOActorImpl::policy_eggroll_logits(
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

void VRPOActorImpl::apply_policy_eggroll_update(const torch::Tensor& delta_weight) {
  policy_lora_->apply_base_weight_update(delta_weight);
}

const LoRALinear& VRPOActorImpl::policy_lora() const {
  return policy_lora_;
}

GoalCritic& VRPOActorImpl::goal_critic() {
  return goal_critic_;
}

std::vector<std::string> VRPOActorImpl::enabled_critic_heads() const {
  return {"extrinsic", "q_critic"};
}

VRPOActor load_vrpo_actor(const std::string& checkpoint_path, const std::string& device) {
  namespace fs = std::filesystem;
  const fs::path base(checkpoint_path);
  const ExperimentConfig config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_inference_checkpoint_metadata(metadata, config);

  torch::Device torch_device(device);
  auto model = VRPOActor(config.model, config.goal_critic, config.es_lora, config.vrpo);
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

VRPOActor clone_vrpo_actor(const VRPOActor& source, const torch::Device& device) {
  if (!source) {
    return nullptr;
  }
  auto clone = VRPOActor(source->config(), source->goal_critic_config(), source->es_lora_config(), source->vrpo_config());
  clone->to(device);
  copy_module_tensors_to(source, clone, device);
  clone->predictor_active_ = source->predictor_active_;
  return clone;
}

void copy_vrpo_actor_tensors_to(const VRPOActor& source, VRPOActor& target, const torch::Device& device) {
  if (!source || !target) {
    throw std::invalid_argument("copy_vrpo_actor_tensors_to requires non-null actors.");
  }
  copy_module_tensors_to(source, target, device);
  target->predictor_active_ = source->predictor_active_;
}

}  // namespace pulsar

#endif
