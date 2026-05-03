#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "pulsar/training/bc_pretrainer.hpp"
#include "test_utils.hpp"

namespace {

pulsar::ExperimentConfig make_bc_smoke_config(const std::filesystem::path& manifest_path) {
  pulsar::ExperimentConfig config;
  config.action_table.builtin = "rlgym_lookup_v1";
  config.model.observation_dim = 132;
  config.model.action_dim = 90;
  config.model.encoder_dim = 16;
  config.model.workspace_dim = 16;
  config.model.stm_slots = 4;
  config.model.stm_key_dim = 8;
  config.model.stm_value_dim = 8;
  config.model.ltm_slots = 4;
  config.model.ltm_dim = 8;
  config.model.controller_dim = 16;
  config.model.value_hidden_dim = 32;
  config.model.value_num_atoms = 51;
  config.model.value_v_min = -10.0F;
  config.model.value_v_max = 10.0F;
  config.ppo.device = "cpu";
  config.ppo.num_envs = 2;
  config.ppo.collection_workers = 0;
  config.ppo.rollout_length = 4;
  config.ppo.minibatch_size = 8;
  config.ppo.update_epochs = 1;
  config.ppo.checkpoint_interval = 1;
  config.ppo.sequence_length = 2;
  config.ppo.burn_in = 0;
  config.offline_dataset.train_manifest = manifest_path.string();
  config.offline_dataset.val_manifest = manifest_path.string();
  config.offline_dataset.batch_size = 16;
  config.behavior_cloning.enabled = true;
  config.behavior_cloning.epochs = 1;
  config.behavior_cloning.sequence_length = 8;
  config.env.seed = 5;
  return config;
}

void write_manifest_fixture(const std::filesystem::path& root) {
  const std::int64_t rows = 64;
  const std::int64_t obs_dim = 132;
  const std::int64_t action_dim = 90;
  torch::Tensor obs = torch::randn({rows, obs_dim});
  torch::Tensor actions = torch::randint(action_dim, {rows}, torch::TensorOptions().dtype(torch::kLong));
  torch::Tensor action_probs = torch::one_hot(actions, action_dim).to(torch::kFloat32);
  torch::Tensor outcome = torch::cat({
      torch::zeros({rows / 2}, torch::TensorOptions().dtype(torch::kLong)),
      torch::ones({rows / 2}, torch::TensorOptions().dtype(torch::kLong)),
  });
  torch::Tensor outcome_known = torch::ones({rows}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor weights = torch::ones({rows});
  torch::Tensor episode_starts = torch::zeros({rows});
  torch::Tensor terminated = torch::zeros({rows});
  torch::Tensor truncated = torch::zeros({rows});
  episode_starts[0] = 1.0F;
  episode_starts[32] = 1.0F;
  terminated[31] = 1.0F;
  terminated[63] = 1.0F;

  std::filesystem::create_directories(root / "data");
  torch::save(obs, (root / "data" / "obs.pt").string());
  torch::save(actions, (root / "data" / "actions.pt").string());
  torch::save(action_probs, (root / "data" / "action_probs.pt").string());
  torch::save(outcome, (root / "data" / "outcome.pt").string());
  torch::save(outcome_known, (root / "data" / "outcome_known.pt").string());
  torch::save(weights, (root / "data" / "weights.pt").string());
  torch::save(episode_starts, (root / "data" / "episode_starts.pt").string());
  torch::save(terminated, (root / "data" / "terminated.pt").string());
  torch::save(truncated, (root / "data" / "truncated.pt").string());

  std::ofstream manifest(root / "data" / "manifest.json");
  manifest << R"({
  "schema_version": 4,
  "observation_dim": 132,
  "action_dim": 90,
  "outcome_classes": 3,
  "shards": [
    {
      "obs_path": "obs.pt",
      "actions_path": "actions.pt",
      "action_probs_path": "action_probs.pt",
      "outcome_path": "outcome.pt",
      "outcome_known_path": "outcome_known.pt",
      "weights_path": "weights.pt",
      "episode_starts_path": "episode_starts.pt",
      "terminated_path": "terminated.pt",
      "truncated_path": "truncated.pt",
      "samples": 64
    }
  ]
})";
}

}  // namespace

int main() {
  try {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "pulsar_bc_offline_test";
    fs::remove_all(root);
    write_manifest_fixture(root);

    pulsar::ExperimentConfig config = make_bc_smoke_config(root / "data" / "manifest.json");
    const auto output_dir = (root / "output").string();
    {
      pulsar::BCPretrainer pretrainer(config);
      pretrainer.train(output_dir);
    }

    pulsar::test::require(
        std::filesystem::exists(root / "output" / "model.pt"),
        "BC training should produce model.pt");
    pulsar::test::require(
        std::filesystem::exists(root / "output" / "metadata.json"),
        "BC training should produce metadata.json");
    pulsar::test::require(
        std::filesystem::exists(root / "output" / "config.json"),
        "BC training should produce config.json");
    pulsar::test::require(
        std::filesystem::exists(root / "output" / "bc_metrics.jsonl"),
        "BC training should produce bc_metrics.jsonl");

    // Verify metrics are sane
    {
      std::ifstream metrics_file(root / "output" / "bc_metrics.jsonl");
      pulsar::test::require(metrics_file.good(), "bc_metrics.jsonl should be readable");
      std::string line;
      bool found_train = false;
      bool found_val = false;
      while (std::getline(metrics_file, line)) {
        if (line.find("\"phase\":\"train\"") != std::string::npos) {
          found_train = true;
          pulsar::test::require(line.find("behavior_samples") != std::string::npos,
                                "train metrics should include behavior_samples");
          pulsar::test::require(line.find("value_samples") != std::string::npos,
                                "train metrics should include value_samples");
        }
        if (line.find("\"phase\":\"val\"") != std::string::npos) {
          found_val = true;
        }
      }
      pulsar::test::require(found_train, "bc_metrics should contain train phase");
      pulsar::test::require(found_val, "bc_metrics should contain val phase");
    }

    // Verify model checkpoint loads and runs forward pass
    {
      auto torch_device = torch::kCPU;
      pulsar::PPOActor loaded = pulsar::load_ppo_actor(root / "output", "cpu");
      pulsar::test::require(loaded.ptr() != nullptr, "load_ppo_actor should return non-null model");
      auto state = loaded->initial_state(2, torch_device);
      auto output = loaded->forward_step(
          torch::randn({2, config.model.observation_dim}, torch_device), std::move(state));
      pulsar::test::require(
          output.policy_logits.sizes() == torch::IntArrayRef({2, config.model.action_dim}),
          "loaded model forward pass should produce valid policy logits");
      pulsar::test::require(
          torch::isfinite(output.policy_logits).all().item<bool>(),
          "loaded model policy logits should be finite");
    }

    fs::remove_all(root);
    std::cout << "pulsar_offline_tests passed\n" << std::flush;
    std::_Exit(EXIT_SUCCESS);
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_offline_tests failed: " << exc.what() << '\n' << std::flush;
    std::_Exit(EXIT_FAILURE);
  }
}
