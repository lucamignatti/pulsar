#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/tracing/tracing.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/ppo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/cuda.h>
#endif

namespace {

#ifdef PULSAR_HAS_TORCH

torch::Device resolve_runtime_device(const std::string& device_name) {
  if (device_name.empty()) {
    if (torch::cuda::is_available()) {
      return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
  }
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

bool should_pin_host_memory(const torch::Device& device) {
#ifdef USE_ROCM
  return false;
#else
  return device.is_cuda();
#endif
}

std::filesystem::path create_seed_checkpoint(
    const pulsar::ExperimentConfig& config,
    const torch::Device& device) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path checkpoint_dir =
      std::filesystem::temp_directory_path() / ("pulsar_bench_seed_" + std::to_string(stamp));
  std::filesystem::create_directories(checkpoint_dir);

  pulsar::SharedActorCritic model(config.model, config.ppo);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  model->to(device);
  normalizer.to(device);

  pulsar::save_experiment_config(config, (checkpoint_dir / "config.json").string());
  pulsar::save_checkpoint_metadata(
      pulsar::CheckpointMetadata{
          .schema_version = config.schema_version,
          .obs_schema_version = config.obs_schema_version,
          .config_hash = pulsar::config_hash(config),
          .action_table_hash = pulsar::action_table_hash(config.action_table),
          .architecture_name = "continuum_dppo",
          .device = config.ppo.device,
          .global_step = 0,
          .update_index = 0,
      },
      (checkpoint_dir / "metadata.json").string());

  torch::serialize::OutputArchive archive;
  model->save(archive);
  normalizer.save(archive);
  archive.save_to((checkpoint_dir / "model.pt").string());
  return checkpoint_dir;
}

#endif

}  // namespace

int main(int argc, char** argv) {
  pulsar::tracing::Session trace_session(
      std::filesystem::current_path() / "pulsar_bench.trace.perfetto.json",
      "pulsar_bench");
  PULSAR_TRACE_SET_THREAD_NAME("main");

  pulsar::ExperimentConfig config;
  const int num_envs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
  const std::size_t collection_workers =
      argc > 2 ? static_cast<std::size_t>(std::max(0, std::atoi(argv[2]))) : 0;
  const bool pin_host_memory = argc > 3 ? std::atoi(argv[3]) != 0 : false;
  const std::string device_name = argc > 4 ? argv[4] : std::string{};
  config.ppo.num_envs = num_envs;
  config.ppo.collection_workers = static_cast<int>(collection_workers);

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  pulsar::BatchedRocketSimCollector collector(
      config,
      obs_builder,
      action_parser,
      done_condition,
      pin_host_memory);
  std::vector<pulsar::ControllerState> actions(
      collector.total_agents(),
      pulsar::ControllerState{.throttle = 1.0F, .boost = true});

  const int warmup_steps = 256;
  const int steps = 10000;

  auto run = [&]() {
    PULSAR_TRACE_SCOPE_CAT("bench", "step_only");
    pulsar::CollectorTimings timings{};
    for (int i = 0; i < warmup_steps; ++i) {
      collector.step(actions, &timings);
    }

    const auto start = std::chrono::steady_clock::now();
    timings = {};
    for (int i = 0; i < steps; ++i) {
      collector.step(actions, &timings);
    }
    return std::pair{
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
        timings,
    };
  };

  const auto [collection_seconds, collection_timings] = run();
  const double step_env_steps_per_second =
      collection_timings.env_step_seconds > 0.0
          ? static_cast<double>(steps * num_envs) / collection_timings.env_step_seconds
          : 0.0;
  const double collection_env_steps_per_second = static_cast<double>(steps * num_envs) / collection_seconds;
  const double collection_agent_steps_per_second =
      collection_env_steps_per_second * static_cast<double>(collector.total_agents()) / static_cast<double>(num_envs);
  const double collection_sim_ticks_per_second =
      collection_env_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double collection_agent_ticks_per_second =
      collection_agent_steps_per_second * static_cast<double>(config.env.tick_skip);

#ifdef PULSAR_HAS_TORCH
  const torch::Device device = resolve_runtime_device(device_name);
  pulsar::TrainerMetrics trainer_metrics{};
  const int trainer_warmup_updates = device.is_cpu() ? 0 : 1;
  const int trainer_updates = device.is_cpu() ? 1 : 2;

  {
    PULSAR_TRACE_SCOPE_CAT("bench", "trainer");
    const std::filesystem::path seed_checkpoint = create_seed_checkpoint(config, device);
    pulsar::ExperimentConfig trainer_config = config;
    trainer_config.ppo.device = device.str();
    trainer_config.ppo.init_checkpoint = seed_checkpoint.string();
    trainer_config.reward.ngp_checkpoint = seed_checkpoint.string();
    trainer_config.reward.ngp_label = "benchmark";
    trainer_config.ppo.self_play.enabled = false;
    trainer_config.wandb.enabled = false;
    trainer_config.reward.online_dataset.enabled = false;
    trainer_config.reward.refresh.enabled = false;

    auto trainer_obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(trainer_config.env);
    auto trainer_action_parser =
        std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(trainer_config.action_table));
    auto trainer_done_condition = std::make_shared<pulsar::SimpleDoneCondition>(trainer_config.env);
    auto trainer_collector = std::make_unique<pulsar::BatchedRocketSimCollector>(
        trainer_config,
        trainer_obs_builder,
        trainer_action_parser,
        trainer_done_condition,
        should_pin_host_memory(device));
    pulsar::PPOTrainer trainer(trainer_config, std::move(trainer_collector), nullptr, seed_checkpoint, false);
    trainer_metrics = trainer.benchmark(trainer_warmup_updates, trainer_updates);
    std::filesystem::remove_all(seed_checkpoint);
  }

  const double agents_per_env = static_cast<double>(collector.total_agents()) / static_cast<double>(num_envs);
  const double trainer_collection_env_steps_per_second =
      trainer_metrics.collection_agent_steps_per_second / agents_per_env;
  const double trainer_update_env_steps_per_second =
      trainer_metrics.update_agent_steps_per_second / agents_per_env;
  const double trainer_overall_env_steps_per_second =
      trainer_metrics.overall_agent_steps_per_second / agents_per_env;
  const double trainer_collection_sim_ticks_per_second =
      trainer_collection_env_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double trainer_collection_agent_ticks_per_second =
      trainer_metrics.collection_agent_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double trainer_overall_sim_ticks_per_second =
      trainer_overall_env_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double trainer_overall_agent_ticks_per_second =
      trainer_metrics.overall_agent_steps_per_second * static_cast<double>(config.env.tick_skip);
#endif

  std::cout << "backend="
#ifdef PULSAR_HAS_ROCKETSIM
            << "rocketsim"
#else
            << "placeholder"
#endif
            << '\n';
#ifdef PULSAR_HAS_TORCH
  std::cout << "device=" << device.str() << '\n';
#else
  std::cout << "device=cpu\n";
#endif
  std::cout << "num_envs=" << num_envs << '\n';
  std::cout << "collection_workers=" << collection_workers << '\n';
  std::cout << "pin_host_memory=" << (pin_host_memory ? 1 : 0) << '\n';
  std::cout << "step_only_env_steps_per_second=" << step_env_steps_per_second << '\n';
  std::cout << "collection_env_steps_per_second=" << collection_env_steps_per_second << '\n';
  std::cout << "collection_agent_steps_per_second=" << collection_agent_steps_per_second << '\n';
  std::cout << "collection_sim_ticks_per_second=" << collection_sim_ticks_per_second << '\n';
  std::cout << "collection_agent_ticks_per_second=" << collection_agent_ticks_per_second << '\n';
#ifdef PULSAR_HAS_TORCH
  std::cout << "trainer_mode=ppo_update\n";
  std::cout << "trainer_minibatch_size=" << config.ppo.minibatch_size << '\n';
  std::cout << "trainer_sequence_length=" << config.ppo.sequence_length << '\n';
  std::cout << "trainer_burn_in=" << config.ppo.burn_in << '\n';
  std::cout << "trainer_epochs=" << config.ppo.epochs << '\n';
  std::cout << "trainer_updates=" << trainer_updates << '\n';
  std::cout << "trainer_rollout_steps=" << (trainer_updates * config.ppo.rollout_length) << '\n';
  std::cout << "trainer_collection_env_steps_per_second=" << trainer_collection_env_steps_per_second << '\n';
  std::cout << "trainer_collection_agent_steps_per_second=" << trainer_metrics.collection_agent_steps_per_second << '\n';
  std::cout << "trainer_collection_sim_ticks_per_second=" << trainer_collection_sim_ticks_per_second << '\n';
  std::cout << "trainer_collection_agent_ticks_per_second=" << trainer_collection_agent_ticks_per_second << '\n';
  std::cout << "trainer_update_env_steps_per_second=" << trainer_update_env_steps_per_second << '\n';
  std::cout << "trainer_update_agent_steps_per_second=" << trainer_metrics.update_agent_steps_per_second << '\n';
  std::cout << "trainer_overall_env_steps_per_second=" << trainer_overall_env_steps_per_second << '\n';
  std::cout << "trainer_overall_agent_steps_per_second=" << trainer_metrics.overall_agent_steps_per_second << '\n';
  std::cout << "trainer_overall_sim_ticks_per_second=" << trainer_overall_sim_ticks_per_second << '\n';
  std::cout << "trainer_overall_agent_ticks_per_second=" << trainer_overall_agent_ticks_per_second << '\n';
  std::cout << "trainer_like_env_steps_per_second=" << trainer_overall_env_steps_per_second << '\n';
  std::cout << "trainer_like_agent_steps_per_second=" << trainer_metrics.overall_agent_steps_per_second << '\n';
  std::cout << "trainer_like_sim_ticks_per_second=" << trainer_overall_sim_ticks_per_second << '\n';
  std::cout << "trainer_like_agent_ticks_per_second=" << trainer_overall_agent_ticks_per_second << '\n';
  std::cout << "trainer_like_steps=" << (trainer_updates * config.ppo.rollout_length) << '\n';
#endif
  std::cout << "agents=" << collector.total_agents() / static_cast<std::size_t>(num_envs) << '\n';
  std::cout << "tick_skip=" << config.env.tick_skip << '\n';
  std::cout << "obs_dim=" << obs_builder->obs_dim() << '\n';
  std::cout << "obs_build_seconds=" << collection_timings.obs_build_seconds << '\n';
  std::cout << "mask_build_seconds=" << collection_timings.mask_build_seconds << '\n';
  std::cout << "env_step_seconds=" << collection_timings.env_step_seconds << '\n';
  std::cout << "done_reset_seconds=" << collection_timings.done_reset_seconds << '\n';
  std::cout << "step_only_env_step_seconds=" << collection_timings.env_step_seconds << '\n';
#ifdef PULSAR_HAS_TORCH
  std::cout << "trainer_obs_build_seconds=" << trainer_metrics.obs_build_seconds << '\n';
  std::cout << "trainer_mask_build_seconds=" << trainer_metrics.mask_build_seconds << '\n';
  std::cout << "trainer_env_step_seconds=" << trainer_metrics.env_step_seconds << '\n';
  std::cout << "trainer_done_reset_seconds=" << trainer_metrics.done_reset_seconds << '\n';
  std::cout << "trainer_policy_forward_seconds=" << trainer_metrics.policy_forward_seconds << '\n';
  std::cout << "trainer_action_decode_seconds=" << trainer_metrics.action_decode_seconds << '\n';
  std::cout << "trainer_reward_model_seconds=" << trainer_metrics.reward_model_seconds << '\n';
  std::cout << "trainer_rollout_append_seconds=" << trainer_metrics.rollout_append_seconds << '\n';
  std::cout << "trainer_gae_seconds=" << trainer_metrics.gae_seconds << '\n';
  std::cout << "trainer_ppo_forward_backward_seconds=" << trainer_metrics.ppo_forward_backward_seconds << '\n';
  std::cout << "trainer_optimizer_step_seconds=" << trainer_metrics.optimizer_step_seconds << '\n';
  std::cout << "trainer_like_obs_build_seconds=" << trainer_metrics.obs_build_seconds << '\n';
  std::cout << "trainer_like_mask_build_seconds=" << trainer_metrics.mask_build_seconds << '\n';
  std::cout << "trainer_like_env_step_seconds=" << trainer_metrics.env_step_seconds << '\n';
  std::cout << "trainer_like_done_reset_seconds=" << trainer_metrics.done_reset_seconds << '\n';
  std::cout << "trainer_like_policy_forward_seconds=" << trainer_metrics.policy_forward_seconds << '\n';
  std::cout << "trainer_like_action_decode_seconds=" << trainer_metrics.action_decode_seconds << '\n';
  std::cout << "trainer_like_reward_model_seconds=" << trainer_metrics.reward_model_seconds << '\n';
#endif
  return 0;
}
