#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "pulsar/training/offline_pretrainer.hpp"

int main() {
  try {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "pulsar_offline_test";
    fs::remove_all(root);
    fs::create_directories(root / "data");

    const int64_t rows = 64;
    const int64_t obs_dim = 8;
    const int64_t action_dim = 4;
    torch::Tensor obs = torch::randn({rows, obs_dim});
    torch::Tensor actions = torch::randint(action_dim, {rows}, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor action_probs = torch::one_hot(actions, action_dim).to(torch::kFloat32);
    torch::Tensor next_goal = torch::randint(3, {rows}, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor weights = torch::ones({rows});
    torch::Tensor episode_starts = torch::zeros({rows});
    episode_starts[0] = 1.0F;
    episode_starts[32] = 1.0F;

    torch::save(obs, (root / "data" / "obs.pt").string());
    torch::save(actions, (root / "data" / "actions.pt").string());
    torch::save(action_probs, (root / "data" / "action_probs.pt").string());
    torch::save(next_goal, (root / "data" / "next_goal.pt").string());
    torch::save(weights, (root / "data" / "weights.pt").string());
    torch::save(episode_starts, (root / "data" / "episode_starts.pt").string());

    std::ofstream manifest(root / "data" / "manifest.json");
    manifest << R"({
  "schema_version": 1,
  "observation_dim": 8,
  "action_dim": 4,
  "next_goal_classes": 3,
  "shards": [
    {
      "obs_path": "obs.pt",
      "actions_path": "actions.pt",
      "action_probs_path": "action_probs.pt",
      "next_goal_path": "next_goal.pt",
      "weights_path": "weights.pt",
      "episode_starts_path": "episode_starts.pt",
      "samples": 64
    }
  ]
})";
    manifest.close();

    pulsar::ExperimentConfig config;
    config.model.observation_dim = static_cast<int>(obs_dim);
    config.model.hidden_sizes = {32, 32};
    config.model.action_dim = static_cast<int>(action_dim);
    config.ppo.device = "cpu";
    config.offline_dataset.train_manifest = (root / "data" / "manifest.json").string();
    config.offline_dataset.val_manifest = (root / "data" / "manifest.json").string();
    config.offline_dataset.batch_size = 16;
    config.offline_dataset.val_batch_size = 32;
    config.behavior_cloning.epochs = 1;
    config.next_goal_predictor.epochs = 1;
    config.next_goal_predictor.hidden_sizes = {16};

    pulsar::OfflinePretrainer pretrainer(config);
    pretrainer.train((root / "output").string());

    if (!fs::exists(root / "output" / "policy" / "model.pt")) {
      throw std::runtime_error("policy checkpoint missing");
    }
    if (!fs::exists(root / "output" / "next_goal" / "model.pt")) {
      throw std::runtime_error("next goal checkpoint missing");
    }

    fs::remove_all(root);
    std::cout << "pulsar_offline_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_offline_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
