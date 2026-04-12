#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>

#include "pulsar/model/actor_critic.hpp"
#include "pulsar/training/ppo_math.hpp"
#include "pulsar/training/rollout_storage.hpp"
#include "test_utils.hpp"

namespace {

void test_masked_sampling_and_log_probs() {
  const torch::Tensor logits = torch::tensor({{1.0F, 10.0F, 2.0F}}, torch::kFloat32);
  const torch::Tensor mask = torch::tensor({{1, 0, 1}}, torch::kBool);
  torch::Tensor log_probs;
  const torch::Tensor deterministic = pulsar::sample_masked_actions(logits, mask, true, &log_probs);
  pulsar::test::require(deterministic.item<std::int64_t>() == 2, "deterministic masked action mismatch");

  for (int i = 0; i < 128; ++i) {
    const torch::Tensor sampled = pulsar::sample_masked_actions(logits, mask, false, nullptr);
    const auto action = sampled.item<std::int64_t>();
    pulsar::test::require(action != 1, "masked action should never be sampled");
  }

  const torch::Tensor expected =
      torch::log_softmax(pulsar::apply_action_mask_to_logits(logits, mask), -1).select(1, 2);
  pulsar::test::require(torch::allclose(log_probs, expected), "masked log-prob mismatch");
}

void test_entropy_projection_and_rollout_storage() {
  const torch::Tensor logits = torch::tensor({{0.0F, 0.0F, 0.0F, 0.0F}}, torch::kFloat32);
  const torch::Tensor half_mask = torch::tensor({{1, 1, 0, 0}}, torch::kBool);
  const torch::Tensor entropy = pulsar::masked_action_entropy(logits, half_mask);
  pulsar::test::require(torch::allclose(entropy, torch::ones_like(entropy), 1.0e-5, 1.0e-5), "normalized entropy should be 1 for uniform valid actions");

  const torch::Tensor projected = pulsar::categorical_value_projection(
      torch::tensor({-10.0F, 0.0F, 10.0F}, torch::kFloat32),
      -10.0F,
      10.0F,
      5);
  pulsar::test::require(torch::allclose(projected.sum(-1), torch::ones({3})), "categorical projection should sum to 1");

  pulsar::RolloutStorage storage(2, 2, 3, 4, torch::kCPU);
  storage.append(
      0,
      torch::ones({2, 3}),
      torch::tensor({1.0F, 1.0F}),
      torch::tensor({{1, 0, 1, 0}, {1, 1, 0, 0}}, torch::kUInt8),
      torch::tensor({1.0F, 0.0F}),
      torch::tensor({2, 1}, torch::kLong),
      torch::tensor({-0.2F, -0.1F}),
      torch::tensor({1.0F, 2.0F}),
      torch::zeros({2}),
      torch::tensor({0.5F, 0.25F}));
  storage.append(
      1,
      torch::zeros({2, 3}),
      torch::zeros({2}),
      torch::tensor({{1, 1, 0, 0}, {1, 0, 0, 1}}, torch::kUInt8),
      torch::tensor({1.0F, 0.0F}),
      torch::tensor({1, 3}, torch::kLong),
      torch::tensor({-0.3F, -0.4F}),
      torch::tensor({0.0F, 0.0F}),
      torch::tensor({1.0F, 1.0F}),
      torch::tensor({0.0F, 0.0F}));
  storage.compute_returns_and_advantages(torch::zeros({2}), 0.99F, 0.95F);
  pulsar::test::require(storage.action_masks[0][0][1].item<std::uint8_t>() == 0, "action mask should round-trip");
  pulsar::test::require(storage.learner_active[0][1].item<float>() == 0.0F, "learner_active should round-trip");
  pulsar::test::require(storage.returns[0][0].item<float>() > 0.0F, "returns should be computed");
}

void test_confidence_weights_adaptive_epsilon_and_precision_validation() {
  pulsar::ModelConfig model_config;
  pulsar::PPOConfig ppo_config;
  ppo_config.value_num_atoms = 11;
  pulsar::SharedActorCritic model(model_config, ppo_config);
  const torch::Tensor value_logits = torch::randn({4, ppo_config.value_num_atoms});

  const torch::Tensor weights = pulsar::compute_confidence_weights(model, ppo_config, value_logits);
  pulsar::test::require(weights.sizes() == torch::IntArrayRef({4}), "confidence weights shape mismatch");
  const torch::Tensor eps = pulsar::compute_adaptive_epsilon(model, ppo_config, value_logits);
  pulsar::test::require(eps.sizes() == torch::IntArrayRef({4}), "adaptive epsilon shape mismatch");

  bool invalid_precision_threw = false;
  try {
    pulsar::validate_precision_mode_or_throw(pulsar::PPOConfig::PrecisionConfig{.mode = "bogus"}, torch::kCPU);
  } catch (const std::runtime_error&) {
    invalid_precision_threw = true;
  }
  pulsar::test::require(invalid_precision_threw, "invalid precision mode should throw");

  bool fp16_cpu_threw = false;
  try {
    pulsar::validate_precision_mode_or_throw(pulsar::PPOConfig::PrecisionConfig{.mode = "amp_fp16"}, torch::kCPU);
  } catch (const std::runtime_error&) {
    fp16_cpu_threw = true;
  }
  pulsar::test::require(fp16_cpu_threw, "amp_fp16 should reject CPU");
}

}  // namespace

int main() {
  try {
    test_masked_sampling_and_log_probs();
    test_entropy_projection_and_rollout_storage();
    test_confidence_weights_adaptive_epsilon_and_precision_validation();
    std::cout << "pulsar_ppo_math_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_ppo_math_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
