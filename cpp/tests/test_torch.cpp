#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/ppo_actor.hpp"

namespace {

pulsar::ModelConfig small_model_config() {
  pulsar::ModelConfig config;
  config.observation_dim = 16;
  config.action_dim = 7;
  config.use_layer_norm = false;
  config.encoder_dim = 8;
  // config.workspace_dim = 8;
  // config.stm_slots = 4;
  // config.stm_key_dim = 4;
  // config.stm_value_dim = 4;
  // config.ltm_slots = 4;
  // config.ltm_dim = 4;
  // config.controller_dim = 8;
  // config.consolidation_stride = 2;
  config.value_hidden_dim = 32;
  return config;
}

pulsar::GoalCriticConfig default_goal_critic_config() {
  pulsar::GoalCriticConfig cfg;
  cfg.embedding_dim = 64;
  cfg.hidden_dim = 256;
  return cfg;
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
      const torch::Tensor loss = output.policy_logits.square().mean()
          + output.value_win_logits.square().mean()
          + output.features.square().mean();
      loss.backward();
      bool saw_encoder_grad = false;
      for (const auto& item : actor->named_parameters(true)) {
        if (item.key().find("encoder.") == 0 && item.value().grad().defined()) {
          saw_encoder_grad = true;
          break;
        }
      }
      if (!saw_encoder_grad) {
        throw std::runtime_error("actor backward did not populate encoder gradients");
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
      pulsar::ModelConfig mamba_config = small_model_config();
      mamba_config.encoder_type = "mamba2";
      mamba_config.encoder_dim = 16;
      mamba_config.num_encoder_blocks = 2;
      mamba_config.transformer_token_group_size = 4;
      pulsar::PPOActor mamba_actor(mamba_config, gc_cfg);
      const auto mamba_step = mamba_actor->forward_step(torch::randn({3, mamba_config.observation_dim}));
      if (mamba_step.policy_logits.sizes() != torch::IntArrayRef({3, mamba_config.action_dim})) {
        throw std::runtime_error("mamba2 policy logits shape mismatch");
      }
      const auto mamba_seq = mamba_actor->forward_sequence(torch::randn({2, 3, mamba_config.observation_dim}));
      if (mamba_seq.policy_logits.sizes() != torch::IntArrayRef({2, 3, mamba_config.action_dim})) {
        throw std::runtime_error("mamba2 sequence logits shape mismatch");
      }
      if (!torch::all(torch::isfinite(mamba_step.features)).item<bool>()) {
        throw std::runtime_error("mamba2 produced non-finite features");
      }
      const torch::Tensor mamba_loss = mamba_step.policy_logits.square().mean()
          + mamba_step.value_win_logits.square().mean()
          + mamba_step.features.square().mean();
      mamba_loss.backward();
      bool saw_feature_scale_grad = false;
      bool saw_decay_grad = false;
      bool saw_conv_grad = false;
      for (const auto& item : mamba_actor->named_parameters(true)) {
        const std::string name = item.key();
        if (name.find("encoder.feature_scale") == 0 && item.value().grad().defined()) {
          saw_feature_scale_grad = true;
        }
        if (name.find("encoder.block_0.decay_bias") == 0 && item.value().grad().defined()) {
          saw_decay_grad = true;
        }
        if (name.find("encoder.block_0.causal_conv") == 0 && item.value().grad().defined()) {
          saw_conv_grad = true;
        }
      }
      if (!saw_feature_scale_grad || !saw_decay_grad || !saw_conv_grad) {
        throw std::runtime_error("mamba2 backward missed expected encoder gradients");
      }
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
