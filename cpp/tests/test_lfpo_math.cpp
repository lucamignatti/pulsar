#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>

#include "pulsar/training/lfpo_math.hpp"
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

void test_latent_relative_advantages_and_clip_loss() {
  const torch::Tensor outcome_logits = torch::tensor(
      {{{{3.0F, 0.0F, -1.0F}, {2.0F, 0.0F, -1.0F}, {1.0F, 0.0F, -1.0F}},
        {{0.0F, 3.0F, -1.0F}, {0.0F, 2.0F, -1.0F}, {0.0F, 1.0F, -1.0F}}}},
      torch::kFloat32);
  const torch::Tensor scores = pulsar::latent_action_scores(outcome_logits);
  pulsar::test::require(scores.sizes() == torch::IntArrayRef({1, 2}), "latent score shape mismatch");
  pulsar::test::require(scores[0][0].item<float>() > scores[0][1].item<float>(), "score action should outrank concede action");

  const torch::Tensor advantages = pulsar::relative_candidate_advantages(scores);
  pulsar::test::require(torch::allclose(advantages.mean(-1), torch::zeros({1}), 1.0e-5, 1.0e-5), "relative advantages should be centered");
  pulsar::test::require(!advantages.requires_grad(), "relative advantages should be detached");

  const torch::Tensor current = torch::log(torch::tensor({{0.8F, 0.2F}}, torch::kFloat32));
  const torch::Tensor old = torch::log(torch::tensor({{0.5F, 0.5F}}, torch::kFloat32));
  const torch::Tensor loss = pulsar::clipped_lfpo_policy_loss(current, old, advantages, 0.2F);
  pulsar::test::require(loss.dim() == 0, "clipped LFPO policy loss should be scalar");
}

void test_entropy_and_rollout_storage() {
  const torch::Tensor logits = torch::tensor({{0.0F, 0.0F, 0.0F, 0.0F}}, torch::kFloat32);
  const torch::Tensor half_mask = torch::tensor({{1, 1, 0, 0}}, torch::kBool);
  const torch::Tensor entropy = pulsar::masked_action_entropy(logits, half_mask);
  pulsar::test::require(
      torch::allclose(entropy, torch::ones_like(entropy), 1.0e-5, 1.0e-5),
      "normalized entropy should be 1 for uniform valid actions");

  pulsar::RolloutStorage storage(2, 2, 3, 4, 2, torch::kCPU);
  storage.append(
      0,
      torch::ones({2, 3}),
      torch::ones({2, 3}),
      torch::tensor({1.0F, 1.0F}),
      torch::tensor({{1, 0, 1, 0}, {1, 1, 0, 0}}, torch::kUInt8),
      torch::tensor({1.0F, 0.0F}),
      torch::tensor({2, 1}, torch::kLong),
      torch::tensor({{2, 0}, {1, 0}}, torch::kLong),
      torch::tensor({{-0.2F, -0.4F}, {-0.1F, -0.7F}}),
      torch::tensor({10, 20}, torch::kLong),
      torch::zeros({2}),
      torch::tensor({2, 2}, torch::kLong));
  storage.append(
      1,
      torch::zeros({2, 3}),
      torch::zeros({2, 3}),
      torch::zeros({2}),
      torch::tensor({{1, 1, 0, 0}, {1, 0, 0, 1}}, torch::kUInt8),
      torch::tensor({1.0F, 0.0F}),
      torch::tensor({1, 3}, torch::kLong),
      torch::tensor({{1, 0}, {3, 0}}, torch::kLong),
      torch::tensor({{-0.3F, -0.5F}, {-0.4F, -0.8F}}),
      torch::tensor({10, 21}, torch::kLong),
      torch::ones({2}),
      torch::tensor({0, 1}, torch::kLong));
  pulsar::test::require(storage.action_masks[0][0][1].item<std::uint8_t>() == 0, "action mask should round-trip");
  pulsar::test::require(storage.candidate_actions[1][0][0].item<std::int64_t>() == 1, "candidate action should round-trip");
  pulsar::test::require(storage.trajectory_ids[1][1].item<std::int64_t>() == 21, "trajectory id should round-trip");
  pulsar::test::require(storage.terminal_outcomes[1][1].item<std::int64_t>() == 1, "terminal outcome should round-trip");
}

}  // namespace

int main() {
  try {
    test_masked_sampling_and_log_probs();
    test_latent_relative_advantages_and_clip_loss();
    test_entropy_and_rollout_storage();
    std::cout << "pulsar_lfpo_math_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_lfpo_math_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
