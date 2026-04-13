#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"

int main() {
  try {
    pulsar::ModelConfig config;
    config.encoder_dim = 16;
    config.use_layer_norm = false;
    pulsar::PPOConfig ppo;
    ppo.value_num_atoms = 31;
    pulsar::SharedActorCritic model(config, ppo);
    auto state = model->initial_state(4, torch::kCPU);
    const auto output = model->forward_step(torch::randn({4, config.observation_dim}), std::move(state));

    if (output.policy_logits.sizes() != torch::IntArrayRef({4, config.action_dim})) {
      throw std::runtime_error("policy logits shape mismatch");
    }
    if (output.value_logits.sizes() != torch::IntArrayRef({4, ppo.value_num_atoms})) {
      throw std::runtime_error("value logits shape mismatch");
    }
    if (output.next_goal_logits.sizes() != torch::IntArrayRef({4, 3})) {
      throw std::runtime_error("next-goal logits shape mismatch");
    }
    if (output.expected_values.sizes() != torch::IntArrayRef({4})) {
      throw std::runtime_error("expected value shape mismatch");
    }

    const auto named_parameters = model->named_parameters(true);
    const auto* first_encoder = named_parameters.find("encoder.0.weight");
    if (first_encoder == nullptr) {
      throw std::runtime_error("encoder projection parameter missing");
    }
    if (first_encoder->sizes() != torch::IntArrayRef({config.encoder_dim, config.observation_dim})) {
      throw std::runtime_error("encoder projection shape mismatch");
    }

    const auto clone = pulsar::clone_shared_model(model, torch::kCPU);
    const auto source_params = model->named_parameters(true);
    const auto clone_params = clone->named_parameters(true);
    for (const auto& item : source_params) {
      if (!torch::allclose(item.value(), clone_params[item.key()])) {
        throw std::runtime_error("cloned model parameters diverged");
      }
    }

    pulsar::ObservationNormalizer normalizer(config.observation_dim);
    normalizer.update(torch::randn({8, config.observation_dim}));
    const auto normalizer_clone = normalizer.clone();
    const torch::Tensor sample = torch::randn({2, config.observation_dim});
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
