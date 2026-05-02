#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "pulsar/training/bc_pretrainer.hpp"

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
    {
      pulsar::BCPretrainer pretrainer(config);
      (void)pretrainer;
    }

    fs::remove_all(root);
    std::cout << "pulsar_offline_tests passed\n" << std::flush;
    std::_Exit(EXIT_SUCCESS);
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_offline_tests failed: " << exc.what() << '\n' << std::flush;
    std::_Exit(EXIT_FAILURE);
  }
}
