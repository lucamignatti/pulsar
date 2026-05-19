#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/mamba2_ops.hpp"
#include "pulsar/model/ppo_actor.hpp"

namespace {

pulsar::ModelConfig small_model_config() {
  pulsar::ModelConfig config;
  config.observation_dim = 16;
  config.action_dim = 7;
  config.use_layer_norm = false;
  config.encoder_dim = 16;
  config.num_encoder_blocks = 2;
  config.sequence_length = 8;
  config.value_hidden_dim = 32;
  return config;
}

pulsar::GoalCriticConfig default_goal_critic_config() {
  pulsar::GoalCriticConfig cfg;
  cfg.embedding_dim = 64;
  cfg.hidden_dim = 256;
  return cfg;
}

torch::Tensor reference_mamba2_scan_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip,
    const torch::Tensor& reset_mask = {}) {
  const auto batch = projected.size(0);
  const auto sequence = projected.size(1);
  const auto embed_dim = projected.size(2) / 5;
  const auto chunks = projected.chunk(5, -1);
  const torch::Tensor x = torch::silu(chunks[0]);
  const torch::Tensor b = torch::sigmoid(chunks[1]);
  const torch::Tensor c = torch::sigmoid(chunks[2]);
  const torch::Tensor z = torch::silu(chunks[3]);
  const torch::Tensor retention =
      torch::sigmoid(chunks[4] + decay_bias.view({1, 1, embed_dim})).clamp(0.01, 0.9999);
  const torch::Tensor recurrent_input = b * x;
  const torch::Tensor reset = reset_mask.defined()
      ? reset_mask.to(projected.device()).to(projected.scalar_type())
      : torch::Tensor{};
  torch::Tensor state = torch::zeros({batch, embed_dim}, projected.options());
  std::vector<torch::Tensor> states;
  states.reserve(static_cast<std::size_t>(sequence));
  for (int64_t t = 0; t < sequence; ++t) {
    if (reset.defined()) {
      state = state * (1.0F - reset.select(1, t)).view({batch, 1});
    }
    state = retention.select(1, t) * state + recurrent_input.select(1, t);
    states.push_back(state);
  }
  const torch::Tensor scanned = torch::stack(states, 1);
  return (c * scanned + skip.view({1, 1, embed_dim}) * x) * z;
}

std::tuple<torch::Tensor, torch::Tensor> reference_mamba2_step_mixed(
    const torch::Tensor& projected,
    const torch::Tensor& previous_scan,
    const torch::Tensor& decay_bias,
    const torch::Tensor& skip) {
  const auto embed_dim = projected.size(1) / 5;
  const auto chunks = projected.chunk(5, -1);
  const torch::Tensor x = torch::silu(chunks[0]);
  const torch::Tensor b = torch::sigmoid(chunks[1]);
  const torch::Tensor c = torch::sigmoid(chunks[2]);
  const torch::Tensor z = torch::silu(chunks[3]);
  const torch::Tensor retention =
      torch::sigmoid(chunks[4] + decay_bias.view({1, embed_dim})).clamp(0.01, 0.9999);
  const torch::Tensor scan = retention * previous_scan + b * x;
  return {(c * scan + skip.view({1, embed_dim}) * x) * z, scan};
}

torch::Tensor reference_causal_conv1d_silu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& reset_mask) {
  const auto batch = input.size(0);
  const auto sequence = input.size(1);
  const auto embed_dim = input.size(2);
  const torch::Tensor reset = reset_mask.to(input.device()).to(input.scalar_type());
  const torch::Tensor zero_step = torch::zeros({batch, 1, embed_dim}, input.options());
  torch::Tensor prev_1 = sequence > 1
      ? torch::cat({zero_step, input.slice(1, 0, sequence - 1)}, 1)
      : torch::zeros_like(input);
  torch::Tensor prev_2 = sequence > 2
      ? torch::cat({zero_step, zero_step, input.slice(1, 0, sequence - 2)}, 1)
      : torch::zeros_like(input);
  torch::Tensor previous_reset = torch::zeros_like(reset);
  if (sequence > 1) {
    previous_reset.slice(1, 1).copy_(reset.slice(1, 0, sequence - 1));
  }
  prev_1 = prev_1 * (1.0F - reset).unsqueeze(-1);
  prev_2 = prev_2 * ((1.0F - reset) * (1.0F - previous_reset)).unsqueeze(-1);
  return torch::silu(
      prev_2 * weight.select(1, 0).view({1, 1, embed_dim})
      + prev_1 * weight.select(1, 1).view({1, 1, embed_dim})
      + input * weight.select(1, 2).view({1, 1, embed_dim})
      + bias.view({1, 1, embed_dim}));
}

}  // namespace

int main() {
  try {
    const pulsar::ModelConfig model_config = small_model_config();
    const pulsar::GoalCriticConfig gc_cfg = default_goal_critic_config();
    pulsar::PPOActor actor(model_config, gc_cfg);
    
    const auto output = actor->forward_step(
        torch::randn({4, model_config.observation_dim}));

    if (output.policy_logits.sizes() != torch::IntArrayRef({4, model_config.action_dim})) {
      throw std::runtime_error("policy logits shape mismatch");
    }
    if (output.value_win_logits.sizes() != torch::IntArrayRef({4, 1})) {
      throw std::runtime_error("value win logits shape mismatch");
    }
    if (output.features.sizes() != torch::IntArrayRef({4, actor->feature_dim()})) {
      throw std::runtime_error("actor feature shape mismatch");
    }
    {
      const torch::Tensor projected = torch::randn({3, 5, 20}, torch::requires_grad());
      const torch::Tensor decay_bias = torch::randn({4}, torch::requires_grad());
      const torch::Tensor skip = torch::randn({4}, torch::requires_grad());
      torch::Tensor reset = torch::zeros({3, 5});
      reset[0][0] = 1.0F;
      reset[1][3] = 1.0F;
      const torch::Tensor actual = pulsar::mamba2_scan_mixed(projected, decay_bias, skip, reset);
      const torch::Tensor expected = reference_mamba2_scan_mixed(projected, decay_bias, skip, reset);
      if (!torch::allclose(actual, expected, 1.0e-6, 1.0e-6)) {
        throw std::runtime_error("mamba2 scan fallback output mismatch");
      }

      const torch::Tensor projected_step = torch::randn({3, 20});
      const torch::Tensor previous_scan = torch::randn({3, 4});
      auto [step_mixed, step_scan] = pulsar::mamba2_step_mixed(projected_step, previous_scan, decay_bias.detach(), skip.detach());
      auto [step_expected, scan_expected] =
          reference_mamba2_step_mixed(projected_step, previous_scan, decay_bias.detach(), skip.detach());
      if (step_mixed.sizes() != torch::IntArrayRef({3, 4}) || step_scan.sizes() != torch::IntArrayRef({3, 4})) {
        throw std::runtime_error("mamba2 step output shape mismatch");
      }
      if (!torch::allclose(step_mixed, step_expected, 1.0e-6, 1.0e-6) ||
          !torch::allclose(step_scan, scan_expected, 1.0e-6, 1.0e-6)) {
        throw std::runtime_error("mamba2 step fallback output mismatch");
      }

    }
    {
      torch::Tensor input = torch::randn({2, 5, 4}, torch::requires_grad());
      torch::Tensor weight = torch::randn({4, 3}, torch::requires_grad());
      torch::Tensor bias = torch::randn({4}, torch::requires_grad());
      torch::Tensor reset = torch::zeros({2, 5});
      reset[0][2] = 1.0F;

      torch::Tensor input_ref = input.detach().clone().set_requires_grad(true);
      torch::Tensor weight_ref = weight.detach().clone().set_requires_grad(true);
      torch::Tensor bias_ref = bias.detach().clone().set_requires_grad(true);
      const torch::Tensor actual = pulsar::mamba2_causal_conv1d_silu(input, weight, bias, reset);
      const torch::Tensor expected = reference_causal_conv1d_silu(input_ref, weight_ref, bias_ref, reset);
      if (!torch::allclose(actual, expected, 1.0e-6, 1.0e-6)) {
        throw std::runtime_error("mamba2 causal conv fallback output mismatch");
      }
      actual.square().mean().backward();
      expected.square().mean().backward();
      if (!torch::allclose(input.grad(), input_ref.grad(), 1.0e-6, 1.0e-6) ||
          !torch::allclose(weight.grad(), weight_ref.grad(), 1.0e-6, 1.0e-6) ||
          !torch::allclose(bias.grad(), bias_ref.grad(), 1.0e-6, 1.0e-6)) {
        throw std::runtime_error("mamba2 causal conv fallback gradient mismatch");
      }
    }

    if (torch::cuda::is_available() && pulsar::mamba2_accelerator_kernels_available()) {
      auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
      torch::Tensor projected = torch::randn({2, 6, 25}, opts).set_requires_grad(true);
      torch::Tensor decay_bias = torch::randn({5}, opts).set_requires_grad(true);
      torch::Tensor skip = torch::randn({5}, opts).set_requires_grad(true);
      torch::Tensor reset = torch::zeros({2, 6}, opts);
      reset[0][2] = 1.0F;
      reset[1][0] = 1.0F;

      torch::Tensor projected_ref = projected.detach().clone().set_requires_grad(true);
      torch::Tensor decay_ref = decay_bias.detach().clone().set_requires_grad(true);
      torch::Tensor skip_ref = skip.detach().clone().set_requires_grad(true);
      const torch::Tensor actual = pulsar::mamba2_scan_mixed(projected, decay_bias, skip, reset);
      const torch::Tensor expected = reference_mamba2_scan_mixed(projected_ref, decay_ref, skip_ref, reset);
      if (!torch::allclose(actual, expected, 1.0e-5, 1.0e-5)) {
        throw std::runtime_error("mamba2 accelerator scan output mismatch");
      }
      actual.square().mean().backward({}, true);
      actual.mean().backward();
      expected.square().mean().backward({}, true);
      expected.mean().backward();
      if (!torch::allclose(projected.grad(), projected_ref.grad(), 1.0e-4, 1.0e-4) ||
          !torch::allclose(decay_bias.grad(), decay_ref.grad(), 1.0e-4, 1.0e-4) ||
          !torch::allclose(skip.grad(), skip_ref.grad(), 1.0e-4, 1.0e-4)) {
        throw std::runtime_error("mamba2 accelerator scan gradient mismatch");
      }

      torch::Tensor extreme_projected = torch::zeros({1, 4, 25}, opts).set_requires_grad(true);
      {
        torch::NoGradGuard no_grad;
        extreme_projected.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 5)}, 80.0F);
        extreme_projected.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(5, 10)}, 80.0F);
        extreme_projected.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(10, 15)}, 80.0F);
        extreme_projected.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(15, 20)}, -80.0F);
        extreme_projected.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(20, 25)}, 80.0F);
      }
      torch::Tensor extreme_decay = torch::full({5}, 80.0F, opts).set_requires_grad(true);
      torch::Tensor extreme_skip = torch::full({5}, 8.0F, opts).set_requires_grad(true);
      torch::Tensor extreme_reset = torch::zeros({1, 4}, opts);
      const torch::Tensor extreme_out =
          pulsar::mamba2_scan_mixed(extreme_projected, extreme_decay, extreme_skip, extreme_reset);
      extreme_out.sum().backward();
      if (!torch::isfinite(extreme_projected.grad()).all().item<bool>() ||
          !torch::isfinite(extreme_decay.grad()).all().item<bool>() ||
          !torch::isfinite(extreme_skip.grad()).all().item<bool>()) {
        throw std::runtime_error("mamba2 accelerator extreme scan produced non-finite gradients");
      }

      torch::Tensor conv_input = torch::randn({2, 6, 5}, opts).set_requires_grad(true);
      torch::Tensor conv_weight = torch::randn({5, 3}, opts).set_requires_grad(true);
      torch::Tensor conv_bias = torch::randn({5}, opts).set_requires_grad(true);
      torch::Tensor conv_reset = torch::zeros({2, 6}, opts);
      conv_reset[0][3] = 1.0F;

      torch::Tensor conv_input_ref = conv_input.detach().clone().set_requires_grad(true);
      torch::Tensor conv_weight_ref = conv_weight.detach().clone().set_requires_grad(true);
      torch::Tensor conv_bias_ref = conv_bias.detach().clone().set_requires_grad(true);
      const torch::Tensor conv_actual =
          pulsar::mamba2_causal_conv1d_silu(conv_input, conv_weight, conv_bias, conv_reset);
      const torch::Tensor conv_expected =
          reference_causal_conv1d_silu(conv_input_ref, conv_weight_ref, conv_bias_ref, conv_reset);
      if (!torch::allclose(conv_actual, conv_expected, 1.0e-5, 1.0e-5)) {
        throw std::runtime_error("mamba2 accelerator causal conv output mismatch");
      }
      conv_actual.square().mean().backward();
      conv_expected.square().mean().backward();
      if (!torch::allclose(conv_input.grad(), conv_input_ref.grad(), 1.0e-4, 1.0e-4) ||
          !torch::allclose(conv_weight.grad(), conv_weight_ref.grad(), 1.0e-4, 1.0e-4) ||
          !torch::allclose(conv_bias.grad(), conv_bias_ref.grad(), 1.0e-4, 1.0e-4)) {
        throw std::runtime_error("mamba2 accelerator causal conv gradient mismatch");
      }
    }

    {
      const torch::Tensor loss = output.policy_logits.square().mean()
          + output.value_win_logits.square().mean()
          + output.features.square().mean();
      loss.backward();
      bool saw_input_projection_grad = false;
      bool saw_decay_grad = false;
      bool saw_conv_grad = false;
      for (const auto& item : actor->named_parameters(true)) {
        const std::string name = item.key();
        if (name.find("encoder.input_projection") == 0 && item.value().grad().defined()) {
          saw_input_projection_grad = true;
        }
        if (name.find("encoder.block_0.decay_bias") == 0 && item.value().grad().defined()) {
          saw_decay_grad = true;
        }
        if (name.find("encoder.block_0.causal_conv") == 0 && item.value().grad().defined()) {
          saw_conv_grad = true;
        }
      }
      if (!saw_input_projection_grad || !saw_decay_grad || !saw_conv_grad) {
        throw std::runtime_error("mamba2 backward missed expected encoder gradients");
      }
      actor->zero_grad();
    }

    {
      const torch::Tensor obs = torch::randn({2, model_config.observation_dim});
      
      
      const auto default_out = actor->forward_step(obs);
      const auto explicit_out = actor->forward_step(
          obs, torch::zeros({2, gc_cfg.goal_dim}));
      if (!torch::allclose(default_out.policy_logits, explicit_out.policy_logits)) {
        throw std::runtime_error("default policy goal should match explicit zero goal");
      }
    }

    {
      const auto seq_out = actor->forward_sequence(torch::randn({2, 3, model_config.observation_dim}));
      if (seq_out.policy_logits.sizes() != torch::IntArrayRef({2, 3, model_config.action_dim})) {
        throw std::runtime_error("sequence logits shape mismatch");
      }
    }

    {
      torch::NoGradGuard no_grad;
      const torch::Tensor obs_seq = torch::randn({4, 3, model_config.observation_dim});
      torch::Tensor episode_starts = torch::zeros({4, 3}, torch::kFloat32);
      episode_starts[0].fill_(1.0F);
      episode_starts[2][1] = 1.0F;
      episode_starts[3][2] = 1.0F;
      const auto sequence_out = actor->forward_sequence(obs_seq, {}, episode_starts);
      torch::Tensor state = actor->initial_recurrent_state(3, torch::kCPU);
      std::vector<torch::Tensor> step_logits;
      for (int t = 0; t < obs_seq.size(0); ++t) {
        torch::Tensor next_state;
        const auto step_out = actor->forward_step_stateful(
            obs_seq[t],
            state,
            episode_starts[t],
            &next_state);
        state = next_state;
        step_logits.push_back(step_out.policy_logits);
      }
      const torch::Tensor stepped_logits = torch::stack(step_logits, 0);
      if (!torch::allclose(sequence_out.policy_logits, stepped_logits, 1.0e-4, 1.0e-4)) {
        throw std::runtime_error("stateful step logits should match sequence logits");
      }
    }

    if (!torch::all(torch::isfinite(output.features)).item<bool>()) {
      throw std::runtime_error("produced non-finite features");
    }

    const auto actor_clone = pulsar::clone_ppo_actor(actor, torch::kCPU);
    const auto source_params = actor->named_parameters(true);
    const auto clone_params = actor_clone->named_parameters(true);
    for (const auto& item : source_params) {
      if (!torch::allclose(item.value(), clone_params[item.key()])) {
        throw std::runtime_error("cloned actor parameters diverged");
      }
    }

    // Goal critic forward pass smoke test
    {
      
      auto out = actor->forward_step(torch::zeros({2, model_config.observation_dim}));
      torch::Tensor goal_score = actor->goal_critic()->forward(
          out.features,
          torch::zeros({2}, torch::TensorOptions().dtype(torch::kLong)),
          torch::zeros({2, gc_cfg.goal_dim}));
      if (goal_score.sizes() != torch::IntArrayRef({2})) {
        throw std::runtime_error("goal critic output shape mismatch");
      }
    }

    // Near-zero goal embeddings should not amplify GCRL gradients by the old
    // hard normalization floor.
    {
      pulsar::GoalCritic critic(8, 4, 64, 32, gc_cfg.goal_dim);
      {
        torch::NoGradGuard no_grad;
        for (torch::Tensor& param : critic->parameters()) {
          param.zero_();
        }
      }
      torch::Tensor embedding = critic->goal_embedding(torch::zeros({4, gc_cfg.goal_dim}));
      embedding.sum().backward();
      double max_abs_grad = 0.0;
      for (const torch::Tensor& param : critic->parameters()) {
        if (!param.grad().defined()) {
          continue;
        }
        max_abs_grad = std::max(max_abs_grad, param.grad().abs().max().item<double>());
      }
      if (!std::isfinite(max_abs_grad) || max_abs_grad > 50.0) {
        throw std::runtime_error("near-zero goal embedding normalization produced excessive gradient");
      }
    }

    // LoRA interface smoke test
    {
      auto lora_params = actor->es_lora_parameters();
      if (lora_params.size() != 2) {
        throw std::runtime_error("LoRA should have A and B parameters");
      }
      auto saved = actor->es_lora_parameters();
      for (auto& p : saved) { p = p.detach().clone(); }

      std::vector<torch::Tensor> perturbation;
      for (const auto& p : lora_params) {
        perturbation.push_back(torch::zeros_like(p));
      }
      actor->apply_lora_perturbation(perturbation, 0.01F);

      actor->restore_es_lora_parameters(saved);

      auto restored = actor->es_lora_parameters();
      for (std::size_t i = 0; i < saved.size(); ++i) {
        if (!torch::allclose(saved[i], restored[i])) {
          throw std::runtime_error("LoRA restore failed");
        }
      }
    }

    {
      pulsar::ESLoraConfig es_cfg;
      es_cfg.rank = 2;
      es_cfg.lora_alpha = 6.0F;
      pulsar::PPOActor custom_actor(model_config, gc_cfg, es_cfg);
      if (custom_actor->policy_lora()->rank() != es_cfg.rank) {
        throw std::runtime_error("configured LoRA rank was not applied");
      }
      if (std::fabs(custom_actor->policy_lora()->scale() - (es_cfg.lora_alpha / static_cast<float>(es_cfg.rank) / 2.0F)) > 1.0e-6F) {
        throw std::runtime_error("configured LoRA alpha was not applied");
      }
    }

    // Policy-head EGGROLL helper smoke test
    {
      
      auto out = actor->forward_step(torch::randn({4, model_config.observation_dim}));
      const int population = 2;
      const int rank = 4;
      const int in_features = actor->policy_lora()->in_features();
      const int out_features = actor->policy_lora()->out_features();
      torch::Tensor A_stack = torch::randn({population, rank, in_features});
      torch::Tensor B_stack = torch::randn({population, out_features, rank});
      torch::Tensor logits = actor->policy_eggroll_logits(out.features, A_stack, B_stack, 0.01F);
      if (logits.sizes() != torch::IntArrayRef({4, model_config.action_dim})) {
        throw std::runtime_error("EGGROLL policy logits shape mismatch");
      }

      torch::Tensor before = actor->policy_lora()->base->weight.detach().clone();
      actor->apply_policy_eggroll_update(torch::ones_like(before) * 0.001F);
      if (torch::allclose(before, actor->policy_lora()->base->weight)) {
        throw std::runtime_error("EGGROLL policy update did not modify base weight");
      }
    }

    pulsar::ObservationNormalizer normalizer(model_config.observation_dim);
    normalizer.update(torch::randn({8, model_config.observation_dim}));
    const auto normalizer_clone = normalizer.clone();
    const torch::Tensor sample = torch::randn({2, model_config.observation_dim});
    if (!torch::allclose(normalizer.normalize(sample), normalizer_clone.normalize(sample))) {
      throw std::runtime_error("normalizer clone mismatch");
    }
    {
      pulsar::ObservationNormalizer left(model_config.observation_dim);
      pulsar::ObservationNormalizer right(model_config.observation_dim);
      const torch::Tensor left_batch = torch::randn({8, model_config.observation_dim}) - 2.0F;
      const torch::Tensor right_batch = torch::randn({12, model_config.observation_dim}) + 3.0F;
      left.update(left_batch);
      right.update(right_batch);
      left.merge(right);

      pulsar::ObservationNormalizer combined(model_config.observation_dim);
      combined.update(torch::cat({left_batch, right_batch}, 0));
      if (!torch::allclose(left.normalize(sample), combined.normalize(sample), 1.0e-5, 1.0e-4)) {
        throw std::runtime_error("normalizer merge mismatch");
      }
    }

    std::cout << "pulsar_torch_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_torch_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
