#include <chrono>
#include <iostream>
#include <memory>

#include "pulsar/config/config.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/rocketsim_engine.hpp"

int main() {
  pulsar::ExperimentConfig config;

  auto reset_mutator = std::make_shared<pulsar::MutatorSequence>(
      std::vector<pulsar::StateMutatorPtr>{
          std::make_shared<pulsar::FixedTeamSizeMutator>(config.env),
          std::make_shared<pulsar::KickoffMutator>(config.env),
      });

  pulsar::RocketSimTransitionEngine engine(config.env, reset_mutator);
  std::vector<pulsar::ControllerState> actions(engine.num_agents(), {.throttle = 1.0F, .boost = true});

  const int warmup_steps = 256;
  for (int i = 0; i < warmup_steps; ++i) {
    engine.step(actions);
  }

  const int steps = 10000;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < steps; ++i) {
    engine.step(actions);
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const double env_steps_per_second = static_cast<double>(steps) / seconds;
  const double agent_steps_per_second = env_steps_per_second * static_cast<double>(engine.num_agents());
  const double sim_ticks_per_second = env_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double agent_ticks_per_second = agent_steps_per_second * static_cast<double>(config.env.tick_skip);

  std::cout << "backend="
#ifdef PULSAR_HAS_ROCKETSIM
            << "rocketsim"
#else
            << "placeholder"
#endif
            << '\n';
  std::cout << "env_steps_per_second=" << env_steps_per_second << '\n';
  std::cout << "agent_steps_per_second=" << agent_steps_per_second << '\n';
  std::cout << "sim_ticks_per_second=" << sim_ticks_per_second << '\n';
  std::cout << "agent_ticks_per_second=" << agent_ticks_per_second << '\n';
  std::cout << "agents=" << engine.num_agents() << '\n';
  std::cout << "tick_skip=" << config.env.tick_skip << '\n';
  return 0;
}
