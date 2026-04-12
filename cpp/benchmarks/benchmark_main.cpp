#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/reward.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"

int main(int argc, char** argv) {
  pulsar::ExperimentConfig config;
  const int num_envs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
  const std::size_t collection_workers =
      argc > 2 ? static_cast<std::size_t>(std::max(1, std::atoi(argv[2]))) : 0;
  const bool pin_host_memory = argc > 3 ? std::atoi(argv[3]) != 0 : false;
  config.ppo.num_envs = num_envs;
  config.ppo.collection_workers = static_cast<int>(collection_workers);

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto reward_fn = std::make_shared<pulsar::CombinedRewardFunction>(config.reward);
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  pulsar::BatchedRocketSimCollector collector(
      config,
      obs_builder,
      action_parser,
      reward_fn,
      done_condition,
      pin_host_memory);
  std::vector<pulsar::ControllerState> actions(
      collector.total_agents(),
      pulsar::ControllerState{.throttle = 1.0F, .boost = true});

  const int warmup_steps = 256;
  const int steps = 10000;

  auto run = [&](bool include_obs, bool include_masks) {
    pulsar::CollectorTimings timings{};
    for (int i = 0; i < warmup_steps; ++i) {
      if (include_obs) {
        collector.collect_observations(&timings);
      }
      if (include_masks) {
        collector.collect_action_masks(&timings);
      }
      collector.step(actions, false, &timings);
    }

    const auto start = std::chrono::steady_clock::now();
    timings = {};
    for (int i = 0; i < steps; ++i) {
      if (include_obs) {
        collector.collect_observations(&timings);
      }
      if (include_masks) {
        collector.collect_action_masks(&timings);
      }
      collector.step(actions, false, &timings);
    }
    return std::pair{
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
        timings,
    };
  };

  const auto [step_seconds, step_timings] = run(false, false);
  const auto [collection_seconds, collection_timings] = run(true, true);
  const double step_env_steps_per_second = static_cast<double>(steps * num_envs) / step_seconds;
  const double collection_env_steps_per_second = static_cast<double>(steps * num_envs) / collection_seconds;
  const double collection_agent_steps_per_second =
      collection_env_steps_per_second * static_cast<double>(collector.total_agents()) / static_cast<double>(num_envs);
  const double collection_sim_ticks_per_second =
      collection_env_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double collection_agent_ticks_per_second =
      collection_agent_steps_per_second * static_cast<double>(config.env.tick_skip);

  std::cout << "backend="
#ifdef PULSAR_HAS_ROCKETSIM
            << "rocketsim"
#else
            << "placeholder"
#endif
            << '\n';
  std::cout << "num_envs=" << num_envs << '\n';
  std::cout << "collection_workers=" << collection_workers << '\n';
  std::cout << "step_only_env_steps_per_second=" << step_env_steps_per_second << '\n';
  std::cout << "collection_env_steps_per_second=" << collection_env_steps_per_second << '\n';
  std::cout << "collection_agent_steps_per_second=" << collection_agent_steps_per_second << '\n';
  std::cout << "collection_sim_ticks_per_second=" << collection_sim_ticks_per_second << '\n';
  std::cout << "collection_agent_ticks_per_second=" << collection_agent_ticks_per_second << '\n';
  std::cout << "agents=" << collector.total_agents() / static_cast<std::size_t>(num_envs) << '\n';
  std::cout << "tick_skip=" << config.env.tick_skip << '\n';
  std::cout << "obs_dim=" << obs_builder->obs_dim() << '\n';
  std::cout << "obs_build_seconds=" << collection_timings.obs_build_seconds << '\n';
  std::cout << "mask_build_seconds=" << collection_timings.mask_build_seconds << '\n';
  std::cout << "env_step_seconds=" << collection_timings.env_step_seconds << '\n';
  std::cout << "reward_done_seconds=" << collection_timings.reward_done_seconds << '\n';
  std::cout << "step_only_env_step_seconds=" << step_timings.env_step_seconds << '\n';
  return 0;
}
