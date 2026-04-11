#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "pulsar/config/config.hpp"
#include "pulsar/core/parallel_executor.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"

int main(int argc, char** argv) {
  pulsar::ExperimentConfig config;
  const int num_envs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
  const std::size_t collection_workers =
      argc > 2 ? static_cast<std::size_t>(std::max(1, std::atoi(argv[2]))) : 0;

  auto reset_mutator = std::make_shared<pulsar::MutatorSequence>(
      std::vector<pulsar::StateMutatorPtr>{
          std::make_shared<pulsar::FixedTeamSizeMutator>(config.env),
          std::make_shared<pulsar::KickoffMutator>(config.env),
      });

  std::vector<std::unique_ptr<pulsar::RocketSimTransitionEngine>> engines;
  engines.reserve(static_cast<std::size_t>(num_envs));
  std::size_t total_agents = 0;
  for (int env_idx = 0; env_idx < num_envs; ++env_idx) {
    pulsar::EnvConfig env_config = config.env;
    env_config.seed += static_cast<std::uint64_t>(env_idx);
    engines.push_back(std::make_unique<pulsar::RocketSimTransitionEngine>(env_config, reset_mutator));
    total_agents += engines.back()->num_agents();
  }

  pulsar::PulsarObsBuilder obs_builder(config.env);
  pulsar::ParallelExecutor executor(collection_workers);
  std::vector<pulsar::ControllerState> actions(engines.front()->num_agents(), {.throttle = 1.0F, .boost = true});
  std::vector<float> obs_buffer(total_agents * obs_builder.obs_dim());

  const int warmup_steps = 256;
  const int steps = 10000;

  auto run = [&](bool include_obs) {
    for (auto& engine : engines) {
      engine->reset(config.env.seed);
    }
    for (int i = 0; i < warmup_steps; ++i) {
      executor.parallel_for(engines.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t engine_idx = begin; engine_idx < end; ++engine_idx) {
          auto& engine = engines[engine_idx];
          engine->step_inplace(actions);
          if (include_obs) {
            const std::size_t offset = engine_idx * static_cast<std::size_t>(engine->num_agents()) * obs_builder.obs_dim();
            obs_builder.build_obs_batch(
                engine->state(),
                std::span<float>(
                    obs_buffer.data() + static_cast<std::ptrdiff_t>(offset),
                    engine->num_agents() * obs_builder.obs_dim()));
          }
        }
      });
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
      executor.parallel_for(engines.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t engine_idx = begin; engine_idx < end; ++engine_idx) {
          auto& engine = engines[engine_idx];
          engine->step_inplace(actions);
          if (include_obs) {
            const std::size_t offset = engine_idx * static_cast<std::size_t>(engine->num_agents()) * obs_builder.obs_dim();
            obs_builder.build_obs_batch(
                engine->state(),
                std::span<float>(
                    obs_buffer.data() + static_cast<std::ptrdiff_t>(offset),
                    engine->num_agents() * obs_builder.obs_dim()));
          }
        }
      });
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };

  const double step_seconds = run(false);
  const double collection_seconds = run(true);
  const double step_env_steps_per_second = static_cast<double>(steps * num_envs) / step_seconds;
  const double collection_env_steps_per_second = static_cast<double>(steps * num_envs) / collection_seconds;
  const double collection_agent_steps_per_second =
      collection_env_steps_per_second * static_cast<double>(engines.front()->num_agents());
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
  std::cout << "collection_workers=" << executor.worker_count() << '\n';
  std::cout << "step_only_env_steps_per_second=" << step_env_steps_per_second << '\n';
  std::cout << "collection_env_steps_per_second=" << collection_env_steps_per_second << '\n';
  std::cout << "collection_agent_steps_per_second=" << collection_agent_steps_per_second << '\n';
  std::cout << "collection_sim_ticks_per_second=" << collection_sim_ticks_per_second << '\n';
  std::cout << "collection_agent_ticks_per_second=" << collection_agent_ticks_per_second << '\n';
  std::cout << "agents=" << engines.front()->num_agents() << '\n';
  std::cout << "tick_skip=" << config.env.tick_skip << '\n';
  std::cout << "obs_dim=" << obs_builder.obs_dim() << '\n';
  return 0;
}
