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
  torch::Tensor next_goal = torch::tensor({0, 1, 2, 0}, torch::TensorOptions().dtype(torch::kLong));
  torch::Tensor weights = torch::tensor({1.0F, 2.0F, 1.0F, 1.0F});
  torch::Tensor episode_starts = torch::tensor({1.0F, 0.0F, 1.0F, 0.0F});

  torch::save(obs, (root / "obs.pt").string());
  torch::save(actions, (root / "actions.pt").string());
  torch::save(action_probs, (root / "action_probs.pt").string());
  torch::save(next_goal, (root / "next_goal.pt").string());
  torch::save(weights, (root / "weights.pt").string());
  if (include_episode_starts) {
    torch::save(episode_starts, (root / "episode_starts.pt").string());
  }

  std::ofstream manifest(root / "manifest.json");
  manifest << "{\n"
              "  \"schema_version\": 3,\n"
              "  \"observation_dim\": 132,\n"
              "  \"action_dim\": 90,\n"
              "  \"next_goal_classes\": 3,\n"
              "  \"shards\": [\n"
              "    {\n"
              "      \"obs_path\": \"obs.pt\",\n"
              "      \"actions_path\": \"actions.pt\",\n"
              "      \"action_probs_path\": \"action_probs.pt\",\n"
              "      \"next_goal_path\": \"next_goal.pt\",\n"
              "      \"weights_path\": \"weights.pt\",\n";
  if (include_episode_starts) {
    manifest << "      \"episode_starts_path\": \"episode_starts.pt\",\n";
  }
  manifest << "      \"samples\": 4\n"
              "    }\n"
              "  ]\n"
              "}\n";
  return root;
}

void test_dataset_iteration_and_trajectories() {
  namespace fs = std::filesystem;
  const fs::path root = make_dataset_fixture(true, false);
  pulsar::OfflineTensorDataset dataset((root / "manifest.json").string());
  pulsar::test::require(dataset.sample_count() == 4, "sample count mismatch");
  pulsar::test::require(dataset.has_episode_starts(), "episode starts should be detected");

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
  fs::remove_all(root);
}

void test_dataset_default_episode_starts_and_mismatch() {
  namespace fs = std::filesystem;
  const fs::path no_starts_root = make_dataset_fixture(false, false);
  pulsar::OfflineTensorDataset no_starts((no_starts_root / "manifest.json").string());
  pulsar::test::require(!no_starts.has_episode_starts(), "episode starts should be absent");
  bool saw_default_start = false;
  no_starts.for_each_batch(4, false, 7, [&](const pulsar::OfflineTensorBatch& batch) {
    saw_default_start = batch.episode_starts[0].item<float>() > 0.5F;
  });
  pulsar::test::require(saw_default_start, "default episode start should be synthesized");

  bool missing_manifest_threw = false;
  try {
    (void)pulsar::load_offline_tensor_manifest((no_starts_root / "missing_manifest.json").string());
  } catch (const std::runtime_error&) {
    missing_manifest_threw = true;
  }
  pulsar::test::require(missing_manifest_threw, "missing manifest should throw an actionable error");

  const fs::path mismatch_root = make_dataset_fixture(true, true);
  pulsar::OfflineTensorDataset mismatch((mismatch_root / "manifest.json").string());
  bool threw = false;
  try {
    mismatch.for_each_batch(4, false, 7, [](const pulsar::OfflineTensorBatch&) {});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  pulsar::test::require(threw, "mismatched leading dimensions should throw");

  fs::remove_all(no_starts_root);
  fs::remove_all(mismatch_root);
}

}  // namespace

int main() {
  try {
    test_dataset_iteration_and_trajectories();
    test_dataset_default_episode_starts_and_mismatch();
    std::cout << "pulsar_dataset_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_dataset_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
