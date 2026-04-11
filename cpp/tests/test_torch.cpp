#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "pulsar/model/actor_critic.hpp"

int main() {
  try {
    pulsar::ModelConfig config;
    pulsar::SharedActorCritic model(config);
    const auto output = model->forward(torch::randn({4, config.observation_dim}));

    if (output.logits.sizes() != torch::IntArrayRef({4, config.action_dim})) {
      throw std::runtime_error("policy logits shape mismatch");
    }
    if (output.values.sizes() != torch::IntArrayRef({4})) {
      throw std::runtime_error("value shape mismatch");
    }

    std::cout << "pulsar_torch_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_torch_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
