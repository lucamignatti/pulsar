#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "pulsar/model/future_evaluator.hpp"
#include "pulsar/model/latent_future_actor.hpp"
#include "pulsar/model/normalizer.hpp"

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
  config.action_embedding_dim = 5;
  config.future_latent_dim = 6;
  config.future_horizon_count = 3;
  return config;
}

pulsar::FutureEvaluatorConfig small_evaluator_config() {
  pulsar::FutureEvaluatorConfig config;
  config.horizons = {1, 2, 3};
  config.latent_dim = 6;
  config.model_dim = 12;
  config.layers = 1;
  config.heads = 3;
  config.feedforward_dim = 24;
  config.outcome_classes = 3;
  return config;
}

}  // namespace

int main() {
  try {
    const pulsar::ModelConfig model_config = small_model_config();
    pulsar::LatentFutureActor actor(model_config);
    auto state = actor->initial_state(4, torch::kCPU);
    const auto output = actor->forward_step(torch::randn({4, model_config.observation_dim}), std::move(state));

    if (output.policy_logits.sizes() != torch::IntArrayRef({4, model_config.action_dim})) {
      throw std::runtime_error("policy logits shape mismatch");
    }
    if (output.features.sizes() != torch::IntArrayRef({4, actor->feature_dim()})) {
      throw std::runtime_error("actor feature shape mismatch");
    }
    const torch::Tensor future = actor->predict_future_latents(output.features, torch::tensor({0, 1, 2, 3}, torch::kLong));
    if (future.sizes() != torch::IntArrayRef({4, model_config.future_horizon_count, model_config.future_latent_dim})) {
      throw std::runtime_error("future latent shape mismatch");
    }

    const auto actor_clone = pulsar::clone_latent_future_actor(actor, torch::kCPU);
    const auto source_params = actor->named_parameters(true);
    const auto clone_params = actor_clone->named_parameters(true);
    for (const auto& item : source_params) {
      if (!torch::allclose(item.value(), clone_params[item.key()])) {
        throw std::runtime_error("cloned actor parameters diverged");
      }
    }

    const pulsar::FutureEvaluatorConfig evaluator_config = small_evaluator_config();
    pulsar::FutureEvaluator evaluator(evaluator_config, model_config.observation_dim);
    const pulsar::FutureEvaluationOutput evaluated =
        evaluator->forward_windows(torch::randn({2, 4, model_config.observation_dim}));
    if (evaluated.embeddings.sizes() != torch::IntArrayRef({2, 3, evaluator_config.latent_dim})) {
      throw std::runtime_error("future evaluator embedding shape mismatch");
    }
    if (evaluated.outcome_logits.sizes() != torch::IntArrayRef({2, 3, evaluator_config.outcome_classes})) {
      throw std::runtime_error("future evaluator outcome shape mismatch");
    }
    if (evaluator->classify_embeddings(torch::randn({2, 5, 3, evaluator_config.latent_dim})).sizes() !=
        torch::IntArrayRef({2, 5, 3, evaluator_config.outcome_classes})) {
      throw std::runtime_error("future evaluator candidate classify shape mismatch");
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
