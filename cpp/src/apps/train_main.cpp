#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/tracing/tracing.hpp"
#include "pulsar/training/appo_trainer.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/self_play_manager.hpp"

namespace {

bool should_pin_host_memory(const std::string& device) {
  return device.rfind("cuda", 0) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pulsar_appo_train <config.json> <checkpoint_dir> [updates]\n"
              << "       omit updates, or pass <=0, to train indefinitely\n";
    return 1;
  }

  try {
    pulsar::tracing::Session trace_session(std::filesystem::path(argv[2]) / "trace.perfetto.json", "pulsar_appo_train");
    PULSAR_TRACE_SET_THREAD_NAME("main");
    const pulsar::ExperimentConfig config = pulsar::load_experiment_config(argv[1]);

    auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
    auto action_parser = std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
    auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
    std::vector<std::unique_ptr<pulsar::BatchedRocketSimCollector>> collectors;
    const int collection_shards = std::max(1, std::min(config.ppo.collection_shards, config.ppo.num_envs));
    collectors.reserve(static_cast<std::size_t>(collection_shards));
    int env_offset = 0;
    for (int shard = 0; shard < collection_shards; ++shard) {
      pulsar::ExperimentConfig shard_config = config;
      const int base_envs = config.ppo.num_envs / collection_shards;
      const int extra_envs = shard < (config.ppo.num_envs % collection_shards) ? 1 : 0;
      shard_config.ppo.num_envs = base_envs + extra_envs;
      if (config.ppo.collection_workers > 0) {
        const int base_workers = config.ppo.collection_workers / collection_shards;
        const int extra_workers = shard < (config.ppo.collection_workers % collection_shards) ? 1 : 0;
        shard_config.ppo.collection_workers = std::max(1, base_workers + extra_workers);
      }
      shard_config.env.seed += static_cast<std::uint64_t>(env_offset);
      collectors.push_back(std::make_unique<pulsar::BatchedRocketSimCollector>(
          shard_config,
          obs_builder,
          action_parser,
          done_condition,
          should_pin_host_memory(config.ppo.device)));
      env_offset += shard_config.ppo.num_envs;
    }

    std::unique_ptr<pulsar::SelfPlayManager> self_play_manager;
    if (config.self_play_league.enabled) {
      self_play_manager = std::make_unique<pulsar::SelfPlayManager>(
          config,
          std::filesystem::path(argv[2]) / "policy_versions",
          obs_builder,
          action_parser,
          torch::Device(config.ppo.device));
    }

    const int updates = argc > 3 ? std::stoi(argv[3]) : 0;
    pulsar::APPOTrainer trainer(
        config,
        std::move(collectors),
        std::move(self_play_manager),
        std::filesystem::path(argv[2]));
    trainer.train(updates, argv[2], argv[1]);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_appo_train failed: " << exc.what() << '\n';
    return 1;
  }
}
