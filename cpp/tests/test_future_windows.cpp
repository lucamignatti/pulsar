#include <cstdlib>
#include <exception>
#include <iostream>

#include "pulsar/training/future_window_builder.hpp"
#include "test_utils.hpp"

namespace {

pulsar::FutureEvaluatorConfig window_config() {
  pulsar::FutureEvaluatorConfig config;
  config.horizons = {1, 3};
  config.latent_dim = 4;
  config.model_dim = 8;
  config.layers = 1;
  config.heads = 2;
  config.feedforward_dim = 16;
  config.outcome_classes = 3;
  return config;
}

pulsar::OfflineTensorPackedBatch make_packed_batch(
    torch::Tensor outcome_known,
    torch::Tensor terminated) {
  pulsar::OfflineTensorPackedBatch batch;
  batch.obs = torch::tensor(
      {{{0.0F, 0.0F}},
       {{1.0F, 1.0F}},
       {{2.0F, 2.0F}},
       {{3.0F, 3.0F}}},
      torch::kFloat32);
  batch.outcome = torch::full({4, 1}, 2, torch::TensorOptions().dtype(torch::kLong));
  batch.outcome_known = outcome_known.reshape({4, 1}).to(torch::kFloat32);
  batch.weights = torch::ones({4, 1});
  batch.episode_starts = torch::zeros({4, 1});
  batch.episode_starts[0][0] = 1.0F;
  batch.terminated = terminated.reshape({4, 1}).to(torch::kFloat32);
  batch.truncated = torch::zeros({4, 1});
  batch.valid_mask = torch::ones({4, 1}, torch::kBool);
  batch.lengths = {4};
  return batch;
}

std::int64_t row_for_time(const pulsar::FutureWindowBatch& batch, std::int64_t t) {
  for (std::int64_t row = 0; row < batch.time_indices.size(0); ++row) {
    if (batch.time_indices[row].item<std::int64_t>() == t) {
      return row;
    }
  }
  throw std::runtime_error("missing future window row");
}

void test_future_masks_do_not_require_outcomes() {
  const auto config = window_config();
  const pulsar::FutureWindowBatch batch = pulsar::build_future_windows_from_packed_batch(
      make_packed_batch(torch::zeros({4}), torch::zeros({4})),
      config);
  const auto row0 = row_for_time(batch, 0);
  const auto row1 = row_for_time(batch, 1);
  pulsar::test::require(batch.future_horizon_mask[row0][0].item<bool>(), "horizon 1 should be valid without outcome");
  pulsar::test::require(batch.future_horizon_mask[row0][1].item<bool>(), "horizon 3 should be valid with context");
  pulsar::test::require(!batch.future_horizon_mask[row1][1].item<bool>(), "horizon crossing past context should be invalid");
  pulsar::test::require(batch.outcome_horizon_mask.sum().item<std::int64_t>() == 0, "unknown outcomes should not train outcome head");
}

void test_terminal_padding_and_unknown_outcome_mask() {
  const auto config = window_config();
  const pulsar::FutureWindowBatch batch = pulsar::build_future_windows_from_packed_batch(
      make_packed_batch(torch::zeros({4}), torch::tensor({0.0F, 1.0F, 0.0F, 0.0F})),
      config);
  const auto row0 = row_for_time(batch, 0);
  pulsar::test::require(batch.future_horizon_mask[row0][1].item<bool>(), "terminal padding should make long horizon valid");
  pulsar::test::require(!batch.outcome_horizon_mask[row0][1].item<bool>(), "unknown terminal label should not train outcome head");
  pulsar::test::require(
      torch::allclose(batch.windows[row0][3], torch::tensor({1.0F, 1.0F})),
      "terminal padding should repeat terminal observation");
}

void test_rollout_episode_boundary_mask() {
  const auto config = window_config();
  pulsar::RolloutWindowSource source;
  source.obs = torch::tensor(
      {{{0.0F, 0.0F}},
       {{1.0F, 1.0F}},
       {{2.0F, 2.0F}},
       {{3.0F, 3.0F}}},
      torch::kFloat32);
  source.final_obs = torch::tensor({{4.0F, 4.0F}}, torch::kFloat32);
  source.terminal_obs = torch::zeros({4, 1, 2});
  source.dones = torch::zeros({4, 1});
  source.trajectory_ids = torch::tensor({{0}, {0}, {1}, {1}}, torch::kLong);
  source.terminal_outcomes = torch::full({4, 1}, 2, torch::TensorOptions().dtype(torch::kLong));

  const pulsar::FutureWindowBatch batch =
      pulsar::build_future_windows_from_rollout(source, config, config.outcome_classes);
  const auto row1 = row_for_time(batch, 1);
  pulsar::test::require(
      !batch.future_horizon_mask[row1][0].item<bool>(),
      "rollout horizon should not cross episode boundary");
}

void test_delta_targets_and_masked_loss() {
  const auto config = window_config();
  const pulsar::FutureWindowBatch batch = pulsar::build_future_windows_from_packed_batch(
      make_packed_batch(torch::ones({4}), torch::zeros({4})),
      config);
  const torch::Tensor targets = pulsar::future_delta_targets(batch.windows, config);
  pulsar::test::require(targets.sizes() == torch::IntArrayRef({batch.windows.size(0), 2, 2}), "delta target shape mismatch");
  const torch::Tensor zero_loss = pulsar::masked_future_delta_loss(
      torch::zeros_like(targets),
      targets,
      torch::zeros_like(batch.future_horizon_mask));
  pulsar::test::require(zero_loss.item<float>() == 0.0F, "zero mask delta loss should be zero");
  const torch::Tensor active_loss = pulsar::masked_future_delta_loss(
      torch::zeros_like(targets),
      targets,
      batch.future_horizon_mask);
  pulsar::test::require(active_loss.item<float>() > 0.0F, "active delta loss should use valid future horizons");
}

}  // namespace

int main() {
  try {
    test_future_masks_do_not_require_outcomes();
    test_terminal_padding_and_unknown_outcome_mask();
    test_rollout_episode_boundary_mask();
    test_delta_targets_and_masked_loss();
    std::cout << "pulsar_future_window_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_future_window_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
