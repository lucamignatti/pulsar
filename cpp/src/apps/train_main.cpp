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
#include "pulsar/training/trainer.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/self_play_manager.hpp"

namespace {

bool should_pin_host_memory(const std::string& device) {
  return device.rfind("cuda", 0) == 0;
}

torch::Device resolve_startup_device(const std::string& device_name) {
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

int team_size_from_mode(const std::string& mode) {
  if (mode == "1v1") return 1;
  if (mode == "2v2") return 2;
  if (mode == "3v3") return 3;
  throw std::invalid_argument("Unknown curriculum mode: " + mode);
}

// Build collectors for a single mode, appending into `collectors`.
// Returns the total number of envs created for this mode.
int build_mode_collectors(
    const pulsar::ExperimentConfig& config,
    const pulsar::CurriculumModeConfig& mode_cfg,
    int total_curriculum_shards,
    std::shared_ptr<pulsar::PulsarObsBuilder> obs_builder,
    std::shared_ptr<pulsar::DiscreteActionParser> action_parser,
    std::shared_ptr<pulsar::SimpleDoneCondition> done_condition,
    bool pin_host,
    std::uint64_t seed_offset,
    std::vector<std::unique_ptr<pulsar::BatchedRocketSimCollector>>& collectors) {

  const int team_size = team_size_from_mode(mode_cfg.mode);
  const int shards = std::max(1, mode_cfg.shards);
  const int num_envs = mode_cfg.num_envs > 0 ? mode_cfg.num_envs : config.ppo.num_envs;

  // Derive worker count: explicit override wins; otherwise take a proportional
  // share of ppo.collection_workers.
  int mode_workers = mode_cfg.collection_workers;
  if (mode_workers <= 0 && config.ppo.collection_workers > 0) {
    mode_workers = std::max(1,
        config.ppo.collection_workers * shards / std::max(1, total_curriculum_shards));
  }

  for (int shard = 0; shard < shards; ++shard) {
    pulsar::ExperimentConfig shard_config = config;
    shard_config.env.team_size = team_size;
    shard_config.env.spawn_opponents = (team_size >= 1);

    const int base_envs = num_envs / shards;
    const int extra_envs = shard < (num_envs % shards) ? 1 : 0;
    shard_config.ppo.num_envs = base_envs + extra_envs;

    if (mode_workers > 0) {
      const int base_workers = mode_workers / shards;
      const int extra_workers = shard < (mode_workers % shards) ? 1 : 0;
      shard_config.ppo.collection_workers = std::max(1, base_workers + extra_workers);
    }

    shard_config.env.seed += seed_offset;

    auto collector = std::make_unique<pulsar::BatchedRocketSimCollector>(
        shard_config, obs_builder, action_parser, done_condition, pin_host);
    collector->set_mode(mode_cfg.mode);
    collectors.push_back(std::move(collector));
    seed_offset += static_cast<std::uint64_t>(shard_config.ppo.num_envs);
  }
  return num_envs;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pulsar_train <config.json> <checkpoint_dir> [updates]\n"
              << "       omit updates, or pass <=0, to train indefinitely\n";
    return 1;
  }

  try {
    pulsar::tracing::Session trace_session(std::filesystem::path(argv[2]) / "trace.perfetto.json", "pulsar_train");
    PULSAR_TRACE_SET_THREAD_NAME("main");
    const pulsar::ExperimentConfig config = pulsar::load_experiment_config(argv[1]);

    // use max team size for the obs builder so observation dimension is
    // constant across all modes (1v1 / 2v2 / 3v3); missing car slots are
    // zero-padded by PulsarObsBuilder
    constexpr int kObsMaxTeamSize = 3;
    auto obs_builder_cfg = config.env;
    obs_builder_cfg.team_size = kObsMaxTeamSize;
    auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(obs_builder_cfg);
    auto action_parser = std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
    auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
    const bool pin_host = should_pin_host_memory(config.ppo.device);

    std::vector<std::unique_ptr<pulsar::BatchedRocketSimCollector>> collectors;

    if (!config.curriculum.modes.empty()) {
      // Multi-mode: one shard group per mode, all collecting concurrently.
      int total_shards = 0;
      for (const auto& m : config.curriculum.modes) {
        total_shards += std::max(1, m.shards);
      }
      collectors.reserve(static_cast<std::size_t>(total_shards));

      std::uint64_t seed_offset = 0;
      for (const auto& mode_cfg : config.curriculum.modes) {
        const int mode_envs = build_mode_collectors(
            config, mode_cfg, total_shards,
            obs_builder, action_parser, done_condition,
            pin_host, seed_offset, collectors);
        seed_offset += static_cast<std::uint64_t>(mode_envs);
        std::cout << "curriculum mode=" << mode_cfg.mode
                  << " shards=" << mode_cfg.shards
                  << " num_envs=" << mode_envs
                  << '\n';
      }
    } else {
      // Single-mode: distribute envs across collection_shards as before.
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
            pin_host));
        env_offset += shard_config.ppo.num_envs;
      }
    }

    std::unique_ptr<pulsar::SelfPlayManager> self_play_manager;
    if (config.self_play_league.enabled) {
      self_play_manager = std::make_unique<pulsar::SelfPlayManager>(
          config,
          std::filesystem::path(argv[2]) / "policy_versions",
          obs_builder,
          action_parser,
          resolve_startup_device(config.ppo.device));
    }

    const int updates = argc > 3 ? std::stoi(argv[3]) : 0;
    pulsar::Trainer trainer(
        config,
        std::move(collectors),
        std::move(self_play_manager),
        std::filesystem::path(argv[2]));
    trainer.train(updates, argv[2], argv[1]);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_train failed: " << exc.what() << '\n';
    return 1;
  }
}
