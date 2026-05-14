#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/appo_trainer.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/self_play_manager.hpp"

namespace {

bool should_pin_host_memory(const std::string& device) {
  return device.rfind("cuda", 0) == 0;
}

int positive_arg_or(int argc, char** argv, int index, int fallback) {
  if (argc <= index) {
    return fallback;
  }
  const int value = std::atoi(argv[index]);
  return value > 0 ? value : fallback;
}

torch::Device resolve_runtime_device(const std::string& device_name) {
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

}  // namespace

int main(int argc, char** argv) {
  const int updates = positive_arg_or(argc, argv, 1, 3);
  const std::filesystem::path config_path =
      argc > 2 ? std::filesystem::path(argv[2]) : std::filesystem::path{"configs/2v2_appo.json"};

  try {
    pulsar::ExperimentConfig config = pulsar::load_experiment_config(config_path.string());
    pulsar::validate_experiment_config(config);
    if (argc > 3) {
      config.ppo.device = resolve_runtime_device(argv[3]).str();
    }
    config.wandb.enabled = false;
    config.ppo.checkpoint_interval = 0;

    constexpr int kObsMaxTeamSize = 3;
    auto obs_builder_cfg = config.env;
    obs_builder_cfg.team_size = kObsMaxTeamSize;
    auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(obs_builder_cfg);
    auto action_parser = std::make_shared<pulsar::DiscreteActionParser>(
        pulsar::ControllerActionTable(config.action_table));
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
    pulsar::APPOTrainer trainer(
        config,
        std::move(collectors),
        std::move(self_play_manager),
        std::filesystem::path{},
        false);

    const pulsar::TrainerBenchmarkMetrics metrics = trainer.benchmark(updates);
    const double total_seconds = std::max(metrics.total_seconds, 1.0e-9);
    const double update_seconds = std::max(metrics.update_seconds, 1.0e-9);
    const double collection_seconds = std::max(metrics.collection_seconds, 1.0e-9);

    std::cout << "config=" << config_path.string() << '\n';
    std::cout << "device=" << config.ppo.device << '\n';
    std::cout << "updates=" << metrics.updates << '\n';
    std::cout << "agent_steps=" << metrics.agent_steps << '\n';
    std::cout << "total_seconds=" << metrics.total_seconds << '\n';
    std::cout << "collection_seconds=" << metrics.collection_seconds << '\n';
    std::cout << "update_seconds=" << metrics.update_seconds << '\n';
    std::cout << "forward_backward_seconds=" << metrics.forward_backward_seconds << '\n';
    std::cout << "optimizer_step_seconds=" << metrics.optimizer_step_seconds << '\n';
    std::cout << "collection_agent_steps_per_second="
              << static_cast<double>(metrics.agent_steps) / collection_seconds << '\n';
    std::cout << "ppo_update_agent_steps_per_second="
              << static_cast<double>(metrics.agent_steps) / update_seconds << '\n';
    std::cout << "overall_agent_steps_per_second="
              << static_cast<double>(metrics.agent_steps) / total_seconds << '\n';
    std::cout << "policy_loss=" << metrics.policy_loss << '\n';
    std::cout << "value_loss=" << metrics.value_loss << '\n';
    std::cout << "entropy=" << metrics.entropy << '\n';
    std::cout << "grad_norm=" << metrics.grad_norm << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_bench failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
