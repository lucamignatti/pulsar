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
  config.workspace_dim = 8;
  config.stm_slots = 4;
  config.stm_key_dim = 4;
  config.stm_value_dim = 4;
  config.ltm_slots = 4;
  config.ltm_dim = 4;
  config.controller_dim = 8;
  config.consolidation_stride = 2;
  config.value_hidden_dim = 32;
  config.value_num_atoms = 51;
  config.value_v_min = -10.0F;
  config.value_v_max = 10.0F;
  return config;
}

pulsar::GoalCriticConfig default_goal_critic_config() {
  pulsar::GoalCriticConfig cfg;
  cfg.num_atoms = 51;
  cfg.v_min = 0.0F;
  cfg.v_max = 25.0F;
  return cfg;
}

}  // namespace

int main() {
  try {
    const pulsar::ModelConfig model_config = small_model_config();
    const pulsar::GoalCriticConfig gc_cfg = default_goal_critic_config();
    pulsar::PPOActor actor(model_config, gc_cfg);
    auto state = actor->initial_state(4, torch::kCPU);
    const auto output = actor->forward_step(
        torch::randn({4, model_config.observation_dim}), std::move(state));

    if (output.policy_logits.sizes() != torch::IntArrayRef({4, model_config.action_dim})) {
      throw std::runtime_error("policy logits shape mismatch");
    }
    if (output.value_win_logits.sizes() != torch::IntArrayRef({4, model_config.value_num_atoms})) {
      throw std::runtime_error("value win logits shape mismatch");
    }
    if (output.features.sizes() != torch::IntArrayRef({4, actor->feature_dim()})) {
      throw std::runtime_error("actor feature shape mismatch");
    }

    const auto actor_clone = pulsar::clone_ppo_actor(actor, torch::kCPU);
    const auto source_params = actor->named_parameters(true);
    const auto clone_params = actor_clone->named_parameters(true);
    for (const auto& item : source_params) {
      if (!torch::allclose(item.value(), clone_params[item.key()])) {
        throw std::runtime_error("cloned actor parameters diverged");
      }
    }

    const torch::Tensor value_support = actor->value_win_support();
    if (value_support.sizes() != torch::IntArrayRef({model_config.value_num_atoms})) {
      throw std::runtime_error("value support shape mismatch");
    }

    const torch::Tensor goal_support = actor->goal_critic_support();
    if (goal_support.sizes() != torch::IntArrayRef({gc_cfg.num_atoms})) {
      throw std::runtime_error("goal critic support shape mismatch");
    }

    // Goal critic forward pass smoke test
    {
      auto s = actor->initial_state(2, torch::kCPU);
      auto out = actor->forward_step(torch::zeros({2, model_config.observation_dim}), std::move(s));
      torch::Tensor goal_logits = actor->goal_critic()->forward(
          out.features,
          torch::zeros({2}, torch::TensorOptions().dtype(torch::kLong)),
          torch::zeros({2}));
      if (goal_logits.sizes() != torch::IntArrayRef({2, gc_cfg.num_atoms})) {
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

    // Policy-head EGGROLL helper smoke test
    {
      auto s = actor->initial_state(4, torch::kCPU);
      auto out = actor->forward_step(torch::randn({4, model_config.observation_dim}), std::move(s));
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

    // Sparse value head forward smoke test
    {
      auto s = actor->initial_state(2, torch::kCPU);
      auto out = actor->forward_step(torch::zeros({2, model_config.observation_dim}), std::move(s));
      const torch::Tensor win_support = actor->value_win_support();
      const float v_min = win_support[0].item<float>();
      const float v_max = win_support[-1].item<float>();
      if (v_min != model_config.value_v_min || v_max != model_config.value_v_max) {
        throw std::runtime_error("value win support range mismatch");
      }
    }

    pulsar::ObservationNormalizer normalizer(model_config.observation_dim);
    normalizer.update(torch::randn({8, model_config.observation_dim}));
    const auto normalizer_clone = normalizer.clone();
    const torch::Tensor sample = torch::randn({2, model_config.observation_dim});
    if (!torch::allclose(normalizer.normalize(sample), normalizer_clone.normalize(sample))) {
      throw std::runtime_error("normalizer clone mismatch");
    }

    std::cout << "pulsar_torch_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_torch_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
