#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/trainer.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"


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

bool parse_bool_override(const std::string& value) {
  if (value == "1" || value == "true" || value == "TRUE" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" || value == "no") {
    return false;
  }
  throw std::invalid_argument("boolean override must be one of 0/1/true/false/yes/no");
}

void apply_benchmark_override(pulsar::ExperimentConfig& config, const std::string& arg) {
  const std::size_t equals = arg.find('=');
  if (equals == std::string::npos || equals == 0 || equals + 1 >= arg.size()) {
    throw std::invalid_argument("benchmark overrides must use key=value syntax: " + arg);
  }
  const std::string key = arg.substr(0, equals);
  const std::string value = arg.substr(equals + 1);
  if (key == "num_envs") {
    config.ppo.num_envs = std::stoi(value);
  } else if (key == "collection_workers") {
    config.ppo.collection_workers = std::stoi(value);
  } else if (key == "collection_shards") {
    config.ppo.collection_shards = std::stoi(value);
  } else if (key == "rollout_length") {
    config.ppo.rollout_length = std::stoi(value);
  } else if (key == "minibatch_size") {
    config.ppo.minibatch_size = std::stoi(value);
  } else if (key == "update_epochs") {
    config.ppo.update_epochs = std::stoi(value);
  } else if (key == "max_forward_samples") {
    config.model.max_forward_samples = std::stoi(value);
  } else if (key == "es_interval") {
    config.es_lora.es_interval = std::stoi(value);
  } else if (key == "es_population_size") {
    config.es_lora.population_size = std::stoi(value);
  } else if (key == "es_virtual_population_waves") {
    config.es_lora.virtual_population_waves = std::stoi(value);
  } else if (key == "es_eval_shards") {
    config.es_lora.eval_shards = std::stoi(value);
  } else if (key == "es_eval_workers") {
    config.es_lora.eval_workers = std::stoi(value);
  } else if (key == "es_eval_num_envs") {
    config.es_lora.eval_num_envs = std::stoi(value);
  } else if (key == "es_eval_rollout_length") {
    config.es_lora.eval_rollout_length = std::stoi(value);
  } else if (key == "es_eval_episodes_per_member") {
    config.es_lora.eval_episodes_per_member = std::stoi(value);
  } else if (key == "es_kl_eval_stride") {
    config.es_lora.kl_eval_stride = std::stoi(value);
  } else if (key == "es_rank_transform") {
    config.es_lora.rank_transform = parse_bool_override(value);
  } else {
    throw std::invalid_argument("unknown benchmark override: " + key);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int updates = positive_arg_or(argc, argv, 1, 3);
  const std::filesystem::path config_path =
      argc > 2 ? std::filesystem::path(argv[2]) : std::filesystem::path{"configs/2v2.json"};

  try {
    pulsar::ExperimentConfig config = pulsar::load_experiment_config(config_path.string());
    int override_start = 3;
    if (argc > 3 && std::string(argv[3]).find('=') == std::string::npos) {
      config.ppo.device = argv[3];
      override_start = 4;
    }
    for (int arg_index = override_start; arg_index < argc; ++arg_index) {
      apply_benchmark_override(config, argv[arg_index]);
    }
    pulsar::validate_experiment_config(config);
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

    pulsar::Trainer trainer(
        config,
        std::move(collectors),
        std::filesystem::path{},
        false);

    const pulsar::TrainerBenchmarkMetrics metrics = trainer.benchmark(updates);
    const double total_seconds = std::max(metrics.total_seconds, 1.0e-9);
    const double update_seconds = std::max(metrics.update_seconds, 1.0e-9);
    const double collection_seconds = std::max(metrics.collection_seconds, 1.0e-9);

    std::cout << "config=" << config_path.string() << '\n';
    std::cout << "device=" << config.ppo.device << '\n';
    std::cout << "model_parameters=" << trainer.model_parameter_count() << '\n';
    std::cout << "num_envs=" << config.ppo.num_envs << '\n';
    std::cout << "collection_shards=" << config.ppo.collection_shards << '\n';
    std::cout << "collection_workers=" << config.ppo.collection_workers << '\n';
    std::cout << "rollout_length=" << config.ppo.rollout_length << '\n';
    std::cout << "minibatch_size=" << config.ppo.minibatch_size << '\n';
    std::cout << "update_epochs=" << config.ppo.update_epochs << '\n';
    std::cout << "max_forward_samples=" << config.model.max_forward_samples << '\n';
    std::cout << "updates=" << metrics.updates << '\n';
    std::cout << "agent_steps=" << metrics.agent_steps << '\n';
    std::cout << "total_seconds=" << metrics.total_seconds << '\n';
    std::cout << "collection_seconds=" << metrics.collection_seconds << '\n';
    std::cout << "update_seconds=" << metrics.update_seconds << '\n';
    std::cout << "collection_obs_build_seconds=" << metrics.obs_build_seconds << '\n';
    std::cout << "collection_mask_build_seconds=" << metrics.mask_build_seconds << '\n';
    std::cout << "collection_policy_forward_seconds=" << metrics.policy_forward_seconds << '\n';
    std::cout << "collection_action_decode_seconds=" << metrics.action_decode_seconds << '\n';
    std::cout << "collection_env_step_seconds=" << metrics.env_step_seconds << '\n';
    std::cout << "collection_done_reset_seconds=" << metrics.done_reset_seconds << '\n';
    std::cout << "forward_backward_seconds=" << metrics.forward_backward_seconds << '\n';
    std::cout << "optimizer_step_seconds=" << metrics.optimizer_step_seconds << '\n';
    std::cout << "collection_agent_steps_per_second="
              << static_cast<double>(metrics.agent_steps) / collection_seconds << '\n';
    std::cout << "ppo_update_agent_steps_per_second="
              << static_cast<double>(metrics.agent_steps) / update_seconds << '\n';
    std::cout << "overall_agent_steps_per_second="
              << static_cast<double>(metrics.agent_steps) / total_seconds << '\n';
    std::cout << "policy_loss=" << metrics.policy_loss << '\n';
    std::cout << "grad_norm=" << metrics.grad_norm << '\n';
    std::cout << "es_updates=" << metrics.es_updates << '\n';
    std::cout << "es_seconds=" << metrics.es_seconds << '\n';
    std::cout << "es_eval_seconds=" << metrics.es_eval_seconds << '\n';
    std::cout << "es_agent_steps_per_second=" << metrics.es_agent_steps_per_second << '\n';
    std::cout << "es_effective_population=" << metrics.es_effective_population << '\n';
    std::cout << "es_virtual_population_waves=" << metrics.es_virtual_population_waves << '\n';
    std::cout << "es_eval_shards=" << metrics.es_eval_shards << '\n';
    std::cout << "es_policy_update_norm=" << metrics.es_policy_update_norm << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_bench failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
