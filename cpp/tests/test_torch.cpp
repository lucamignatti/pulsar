#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "pulsar/model/actor_critic.hpp"

int main() {
  try {
    pulsar::ModelConfig config;
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

    std::cout << "pulsar_torch_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_torch_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
