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

}  // namespace

int main() {
  try {
    const pulsar::ModelConfig model_config = small_model_config();
    pulsar::PPOActor actor(model_config);
    auto state = actor->initial_state(4, torch::kCPU);
    const auto output = actor->forward_step(torch::randn({4, model_config.observation_dim}), std::move(state));

    if (output.policy_logits.sizes() != torch::IntArrayRef({4, model_config.action_dim})) {
      throw std::runtime_error("policy logits shape mismatch");
    }
    if (output.value_ext.logits.sizes() != torch::IntArrayRef({4, model_config.value_num_atoms})) {
      throw std::runtime_error("value logits shape mismatch");
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

    const torch::Tensor value_support = actor->value_support("extrinsic");
    if (value_support.sizes() != torch::IntArrayRef({model_config.value_num_atoms})) {
      throw std::runtime_error("value support shape mismatch");
    }

    // Smoke test: disabled critic heads must not crash forward pass.
    {
      pulsar::ModelConfig cfg = small_model_config();
      pulsar::CriticConfig critic;
      critic.controllability.enabled = false;
      pulsar::PPOActor actor_with_disabled(cfg, critic);
      auto s = actor_with_disabled->initial_state(2, torch::kCPU);
      auto obs = torch::zeros({2, cfg.observation_dim});
      auto starts = torch::zeros({2});
      auto out = actor_with_disabled->forward_step(obs, std::move(s), starts);

      const auto enabled = actor_with_disabled->enabled_critic_heads();
      bool found_ctrl = false;
      for (const auto& h : enabled) {
        if (h == "controllability") found_ctrl = true;
      }
      if (found_ctrl) {
        throw std::runtime_error("controllability should not appear in enabled_critic_heads when disabled");
      }
      if (out.value_ctrl.logits.defined() && out.value_ctrl.logits.numel() == 0) {
        throw std::runtime_error("disabled value head should still produce output");
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
