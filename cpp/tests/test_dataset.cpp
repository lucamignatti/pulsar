#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "pulsar/training/offline_dataset.hpp"
#include "test_utils.hpp"

namespace {

std::filesystem::path make_dataset_fixture(bool include_episode_starts, bool mismatched_actions) {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / (include_episode_starts ? "pulsar_dataset_fixture" : "pulsar_dataset_no_starts");
  fs::remove_all(root);
  fs::create_directories(root);

  torch::Tensor obs = torch::randn({4, 132});
  torch::Tensor actions = torch::tensor({0, 1, 2, 3}, torch::TensorOptions().dtype(torch::kLong));
  if (mismatched_actions) {
    actions = torch::tensor({0, 1, 2}, torch::TensorOptions().dtype(torch::kLong));
  }
  torch::Tensor action_probs = torch::one_hot(torch::tensor({0, 1, 2, 3}), 90).to(torch::kFloat32);
  torch::Tensor outcome = torch::tensor({0, 1, 2, 0}, torch::TensorOptions().dtype(torch::kLong));
  torch::Tensor outcome_known = torch::ones({4}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor weights = torch::tensor({1.0F, 2.0F, 1.0F, 1.0F});
  torch::Tensor episode_starts = torch::tensor({1.0F, 0.0F, 1.0F, 0.0F});
  torch::Tensor terminated = torch::tensor({0.0F, 1.0F, 0.0F, 1.0F});
  torch::Tensor truncated = torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32));

  torch::save(obs, (root / "obs.pt").string());
  torch::save(actions, (root / "actions.pt").string());
  torch::save(action_probs, (root / "action_probs.pt").string());
  torch::save(outcome, (root / "outcome.pt").string());
  torch::save(outcome_known, (root / "outcome_known.pt").string());
  torch::save(weights, (root / "weights.pt").string());
  if (include_episode_starts) {
    torch::save(episode_starts, (root / "episode_starts.pt").string());
  }
  torch::save(terminated, (root / "terminated.pt").string());
  torch::save(truncated, (root / "truncated.pt").string());

  std::ofstream manifest(root / "manifest.json");
  manifest << "{\n"
              "  \"schema_version\": 4,\n"
              "  \"observation_dim\": 132,\n"
              "  \"action_dim\": 90,\n"
              "  \"outcome_classes\": 3,\n"
              "  \"shards\": [\n"
              "    {\n"
              "      \"obs_path\": \"obs.pt\",\n"
              "      \"actions_path\": \"actions.pt\",\n"
              "      \"action_probs_path\": \"action_probs.pt\",\n"
              "      \"outcome_path\": \"outcome.pt\",\n"
              "      \"outcome_known_path\": \"outcome_known.pt\",\n"
              "      \"weights_path\": \"weights.pt\",\n";
  if (include_episode_starts) {
    manifest << "      \"episode_starts_path\": \"episode_starts.pt\",\n";
  }
  manifest << "      \"terminated_path\": \"terminated.pt\",\n"
              "      \"truncated_path\": \"truncated.pt\",\n";
  manifest << "      \"samples\": 4\n"
              "    }\n"
              "  ]\n"
              "}\n";
  return root;
}

void test_dataset_iteration_and_trajectories() {
  namespace fs = std::filesystem;
  const fs::path root = make_dataset_fixture(true, false);
  {
    pulsar::OfflineTensorDataset dataset((root / "manifest.json").string());
    pulsar::test::require(dataset.sample_count() == 4, "sample count mismatch");
    pulsar::test::require(dataset.has_episode_starts(), "episode starts should be detected");
    pulsar::test::require(dataset.has_trajectory_end_flags(), "trajectory end flags should be detected");

    std::int64_t seen_rows = 0;
    dataset.for_each_batch(2, false, 7, [&](const pulsar::OfflineTensorBatch& batch) {
      seen_rows += batch.obs.size(0);
      pulsar::test::require(batch.obs.size(1) == 132, "batch obs dim mismatch");
      pulsar::test::require(batch.actions.defined(), "actions should be defined");
    });
    pulsar::test::require(seen_rows == 4, "for_each_batch row count mismatch");

    int trajectories = 0;
    dataset.for_each_trajectory(false, 7, [&](const pulsar::OfflineTensorBatch& batch) {
      ++trajectories;
      pulsar::test::require(batch.episode_starts[0].item<float>() > 0.5F, "trajectory should start with episode_starts");
    });
    pulsar::test::require(trajectories == 2, "trajectory segmentation mismatch");

    int packed_batches = 0;
    dataset.for_each_packed_trajectory_batch(4, false, 7, [&](const pulsar::OfflineTensorPackedBatch& batch) {
      ++packed_batches;
      pulsar::test::require(batch.obs.size(0) == 2, "packed batch time dimension mismatch");
      pulsar::test::require(batch.obs.size(1) == 2, "packed batch count mismatch");
      pulsar::test::require(batch.obs.size(2) == 132, "packed batch obs dim mismatch");
      pulsar::test::require(batch.valid_mask.all().item<bool>(), "packed batch should be fully valid in this fixture");
    });
    pulsar::test::require(packed_batches == 1, "packed trajectory batching mismatch");
  }
  fs::remove_all(root);
}

void test_dataset_default_episode_starts_and_mismatch() {
  namespace fs = std::filesystem;
  const fs::path no_starts_root = make_dataset_fixture(false, false);
  {
    pulsar::OfflineTensorDataset no_starts((no_starts_root / "manifest.json").string());
    pulsar::test::require(!no_starts.has_episode_starts(), "episode starts should be absent");
    pulsar::test::require(no_starts.has_trajectory_end_flags(), "trajectory end flags should still be present");
    bool saw_default_start = false;
    no_starts.for_each_batch(4, false, 7, [&](const pulsar::OfflineTensorBatch& batch) {
      saw_default_start = batch.episode_starts[0].item<float>() > 0.5F;
    });
    pulsar::test::require(saw_default_start, "default episode start should be synthesized");
  }

  bool missing_manifest_threw = false;
  try {
    (void)pulsar::load_offline_tensor_manifest((no_starts_root / "missing_manifest.json").string());
  } catch (const std::runtime_error&) {
    missing_manifest_threw = true;
  }
  pulsar::test::require(missing_manifest_threw, "missing manifest should throw an actionable error");

  const fs::path mismatch_root = make_dataset_fixture(true, true);
  {
    pulsar::OfflineTensorDataset mismatch((mismatch_root / "manifest.json").string());
    bool threw = false;
    try {
      mismatch.for_each_batch(4, false, 7, [](const pulsar::OfflineTensorBatch&) {});
    } catch (const std::runtime_error&) {
      threw = true;
    }
    pulsar::test::require(threw, "mismatched leading dimensions should throw");
  }

  fs::remove_all(no_starts_root);
  fs::remove_all(mismatch_root);
}

void test_packed_batch_chunking() {
  namespace fs = std::filesystem;
  // Create a fixture with 100 samples in a single trajectory
  const fs::path root = fs::temp_directory_path() / "pulsar_dataset_chunk_test";
  fs::remove_all(root);
  fs::create_directories(root);

  const int64_t samples = 100;
  const int64_t obs_dim = 4;
  const int64_t action_dim = 7;
  torch::Tensor obs = torch::randn({samples, obs_dim});
  torch::Tensor actions = torch::randint(0, action_dim, {samples}, torch::kLong);
  torch::Tensor action_probs = torch::rand({samples, action_dim});
  action_probs = action_probs / action_probs.sum(-1, true);
  torch::Tensor outcome = torch::randint(0, 3, {samples}, torch::kLong);
  torch::Tensor outcome_known = torch::ones({samples}, torch::kFloat32);
  torch::Tensor weights = torch::ones({samples}, torch::kFloat32);
  torch::Tensor episode_starts = torch::zeros({samples}, torch::kFloat32);
  episode_starts[0] = 1.0F;
  torch::Tensor terminated = torch::cat({
      torch::zeros({samples - 1}, torch::kFloat32),
      torch::ones({1}, torch::kFloat32),
  });
  torch::Tensor truncated = torch::zeros({samples}, torch::kFloat32);

  torch::save(obs, (root / "obs.pt").string());
  torch::save(actions, (root / "actions.pt").string());
  torch::save(action_probs, (root / "action_probs.pt").string());
  torch::save(outcome, (root / "outcome.pt").string());
  torch::save(outcome_known, (root / "outcome_known.pt").string());
  torch::save(weights, (root / "weights.pt").string());
  torch::save(episode_starts, (root / "episode_starts.pt").string());
  torch::save(terminated, (root / "terminated.pt").string());
  torch::save(truncated, (root / "truncated.pt").string());

  std::ofstream manifest(root / "manifest.json");
  manifest << "{\n"
              "  \"schema_version\": 4,\n"
              "  \"observation_dim\": " << obs_dim << ",\n"
              "  \"action_dim\": " << action_dim << ",\n"
              "  \"outcome_classes\": 3,\n"
              "  \"shards\": [\n"
              "    {\n"
              "      \"obs_path\": \"obs.pt\",\n"
              "      \"actions_path\": \"actions.pt\",\n"
              "      \"action_probs_path\": \"action_probs.pt\",\n"
              "      \"outcome_path\": \"outcome.pt\",\n"
              "      \"outcome_known_path\": \"outcome_known.pt\",\n"
              "      \"weights_path\": \"weights.pt\",\n"
              "      \"episode_starts_path\": \"episode_starts.pt\",\n"
              "      \"terminated_path\": \"terminated.pt\",\n"
              "      \"truncated_path\": \"truncated.pt\",\n"
              "      \"samples\": " << samples << "\n"
              "    }\n"
              "  ]\n"
              "}\n";

  pulsar::OfflineTensorDataset dataset((root / "manifest.json").string());
  int total_valid = 0;
  int chunk_count = 0;
  std::vector<int> chunk_sizes;
  dataset.for_each_packed_trajectory_batch_until(
      1000, false, 7,
      [&](const pulsar::OfflineTensorPackedBatch& batch) -> bool {
        ++chunk_count;
        auto n_valid = batch.valid_mask.sum().item<int64_t>();
        total_valid += static_cast<int>(n_valid);
        chunk_sizes.push_back(static_cast<int>(n_valid));
        return true;
      },
      32);
  pulsar::test::require(total_valid == 100,
      "packed batch chunking total samples mismatch: " + std::to_string(total_valid));
  pulsar::test::require(chunk_sizes == std::vector<int>({32, 32, 32, 4}),
      "packed batch chunk sizes incorrect: got " +
      std::to_string(chunk_sizes[0]) + "," +
      std::to_string(chunk_sizes[1]) + "," +
      std::to_string(chunk_sizes[2]) + "," +
      std::to_string(chunk_sizes[3]));

  // Verify chunk reset: first chunk should have episode_starts at position 0
  bool saw_chunk_reset = false;
  dataset.for_each_packed_trajectory_batch(
      1000, false, 7,
      [&](const pulsar::OfflineTensorPackedBatch& batch) {
        if (batch.episode_starts[0].all().item<bool>()) {
          saw_chunk_reset = true;
        }
      },
      32);
  pulsar::test::require(saw_chunk_reset,
      "chunk reset: first position should have episode_starts all true");

  fs::remove_all(root);
}

void test_outcome_filtering() {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "pulsar_dataset_outcome_test";
  fs::remove_all(root);
  fs::create_directories(root);

  int64_t samples = 4;
  torch::Tensor obs = torch::randn({samples, 4});
  torch::Tensor actions = torch::randint(0, 7, {samples}, torch::kLong);
  torch::Tensor action_probs = torch::rand({samples, 7});
  action_probs = action_probs / action_probs.sum(-1, true);
  // outcome = [0, 1, 2, 0], outcome_known = [1, 0, 0, 1] => only 2 known outcomes
  torch::Tensor outcome = torch::tensor({static_cast<int64_t>(0), 1, 2, 0}, torch::kLong);
  torch::Tensor outcome_known = torch::tensor({1.0F, 0.0F, 0.0F, 1.0F});
  torch::Tensor weights = torch::tensor({1.0F, 2.0F, 1.0F, 3.0F});
  torch::Tensor episode_starts = torch::tensor({1.0F, 0.0F, 1.0F, 0.0F});
  torch::Tensor terminated = torch::tensor({0.0F, 1.0F, 0.0F, 1.0F});
  torch::Tensor truncated = torch::zeros({samples});

  torch::save(obs, (root / "obs.pt").string());
  torch::save(actions, (root / "actions.pt").string());
  torch::save(action_probs, (root / "action_probs.pt").string());
  torch::save(outcome, (root / "outcome.pt").string());
  torch::save(outcome_known, (root / "outcome_known.pt").string());
  torch::save(weights, (root / "weights.pt").string());
  torch::save(episode_starts, (root / "episode_starts.pt").string());
  torch::save(terminated, (root / "terminated.pt").string());
  torch::save(truncated, (root / "truncated.pt").string());

  std::ofstream manifest(root / "manifest.json");
  manifest << "{\n"
              "  \"schema_version\": 4,\n"
              "  \"observation_dim\": 4,\n"
              "  \"action_dim\": 7,\n"
              "  \"outcome_classes\": 3,\n"
              "  \"shards\": [\n"
              "    {\n"
              "      \"obs_path\": \"obs.pt\",\n"
              "      \"actions_path\": \"actions.pt\",\n"
              "      \"action_probs_path\": \"action_probs.pt\",\n"
              "      \"outcome_path\": \"outcome.pt\",\n"
              "      \"outcome_known_path\": \"outcome_known.pt\",\n"
              "      \"weights_path\": \"weights.pt\",\n"
              "      \"episode_starts_path\": \"episode_starts.pt\",\n"
              "      \"terminated_path\": \"terminated.pt\",\n"
              "      \"truncated_path\": \"truncated.pt\",\n"
              "      \"samples\": " << samples << "\n"
              "    }\n"
              "  ]\n"
              "}\n";

  pulsar::OfflineTensorDataset dataset((root / "manifest.json").string());
  int known_count = 0;
  int total_samples = 0;
  dataset.for_each_trajectory(false, 7, [&](const pulsar::OfflineTensorBatch& batch) {
    auto known = batch.outcome_known.greater(0.5F);
    known_count += static_cast<int>(known.sum().item<int64_t>());
    total_samples += static_cast<int>(batch.outcome.size(0));
  });
  pulsar::test::require(known_count == 2,
      "outcome_known filtering: expected 2 known outcomes, got " + std::to_string(known_count));
  pulsar::test::require(total_samples == 4,
      "outcome_known filtering: total samples unchanged");

  fs::remove_all(root);
}

void test_sample_weights() {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "pulsar_dataset_weight_test";
  fs::remove_all(root);
  fs::create_directories(root);

  const int64_t samples = 4;
  torch::Tensor obs = torch::randn({samples, 8});
  torch::Tensor actions = torch::randint(0, 7, {samples}, torch::kLong);
  torch::Tensor action_probs = torch::rand({samples, 7});
  action_probs = action_probs / action_probs.sum(-1, true);
  torch::Tensor outcome = torch::randint(0, 3, {samples}, torch::kLong);
  torch::Tensor outcome_known = torch::ones({samples});
  // Two different weight values
  torch::Tensor weights = torch::tensor({1.0F, 10.0F, 1.0F, 10.0F});
  torch::Tensor episode_starts = torch::ones({samples});
  torch::Tensor terminated = torch::ones({samples});
  torch::Tensor truncated = torch::zeros({samples});

  torch::save(obs, (root / "obs.pt").string());
  torch::save(actions, (root / "actions.pt").string());
  torch::save(action_probs, (root / "action_probs.pt").string());
  torch::save(outcome, (root / "outcome.pt").string());
  torch::save(outcome_known, (root / "outcome_known.pt").string());
  torch::save(weights, (root / "weights.pt").string());
  torch::save(episode_starts, (root / "episode_starts.pt").string());
  torch::save(terminated, (root / "terminated.pt").string());
  torch::save(truncated, (root / "truncated.pt").string());

  std::ofstream manifest(root / "manifest.json");
  manifest << "{\n"
              "  \"schema_version\": 4,\n"
              "  \"observation_dim\": 8,\n"
              "  \"action_dim\": 7,\n"
              "  \"outcome_classes\": 3,\n"
              "  \"shards\": [\n"
              "    {\n"
              "      \"obs_path\": \"obs.pt\",\n"
              "      \"actions_path\": \"actions.pt\",\n"
              "      \"action_probs_path\": \"action_probs.pt\",\n"
              "      \"outcome_path\": \"outcome.pt\",\n"
              "      \"outcome_known_path\": \"outcome_known.pt\",\n"
              "      \"weights_path\": \"weights.pt\",\n"
              "      \"episode_starts_path\": \"episode_starts.pt\",\n"
              "      \"terminated_path\": \"terminated.pt\",\n"
              "      \"truncated_path\": \"truncated.pt\",\n"
              "      \"samples\": " << samples << "\n"
              "    }\n"
              "  ]\n"
              "}\n";

  // Verify weights are loaded correctly and vary
  pulsar::OfflineTensorDataset dataset((root / "manifest.json").string());
  bool found_light = false;
  bool found_heavy = false;
  dataset.for_each_batch(samples, false, 7, [&](const pulsar::OfflineTensorBatch& batch) {
    for (int i = 0; i < batch.weights.size(0); ++i) {
      float w = batch.weights[i].item<float>();
      if (w < 2.0F) found_light = true;
      if (w > 5.0F) found_heavy = true;
    }
  });
  pulsar::test::require(found_light && found_heavy,
      "sample weights should have distinct values loaded from fixture");

  fs::remove_all(root);
}

}  // namespace

int main() {
  try {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    test_dataset_iteration_and_trajectories();
    test_dataset_default_episode_starts_and_mismatch();
    test_packed_batch_chunking();
    test_outcome_filtering();
    test_sample_weights();
    std::cout << "pulsar_dataset_tests passed\n" << std::flush;
    std::_Exit(EXIT_SUCCESS);
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_dataset_tests failed: " << exc.what() << '\n' << std::flush;
    std::_Exit(EXIT_FAILURE);
  }
}
