#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/reward.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/ppo_trainer.hpp"
#include "pulsar/training/self_play_manager.hpp"

namespace {

bool should_pin_host_memory(const std::string& device) {
  if (device == "cpu") {
    return false;
  }
#ifdef USE_ROCM
  return false;
#else
  return true;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pulsar_train <config.json> <checkpoint_dir> [updates]\n";
    return 1;
  }

  const pulsar::ExperimentConfig config = pulsar::load_experiment_config(argv[1]);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser = std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto reward_fn = std::make_shared<pulsar::CombinedRewardFunction>(config.reward);
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  auto collector = std::make_unique<pulsar::BatchedRocketSimCollector>(
      config,
      obs_builder,
      action_parser,
      reward_fn,
      done_condition,
      should_pin_host_memory(config.ppo.device));

  std::unique_ptr<pulsar::SelfPlayManager> self_play_manager;
  if (config.ppo.self_play.enabled) {
    self_play_manager = std::make_unique<pulsar::SelfPlayManager>(
        config,
        std::filesystem::path(argv[2]) / "policy_versions",
        obs_builder,
        action_parser,
        torch::Device(config.ppo.device));
  }

  const int updates = argc > 3 ? std::stoi(argv[3]) : 100;
  pulsar::PPOTrainer trainer(
      config,
      std::move(collector),
      action_parser,
      std::move(self_play_manager));
  trainer.train(updates, argv[2], argv[1]);
  return 0;
}
