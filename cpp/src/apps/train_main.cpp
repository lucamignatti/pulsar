#include <iostream>
#include <memory>
#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/reward.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/ppo_trainer.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pulsar_train <config.json> <checkpoint_dir> [updates]\n";
    return 1;
  }

  const pulsar::ExperimentConfig config = pulsar::load_experiment_config(argv[1]);

  auto reset_mutator = std::make_shared<pulsar::MutatorSequence>(
      std::vector<pulsar::StateMutatorPtr>{
          std::make_shared<pulsar::FixedTeamSizeMutator>(config.env),
          std::make_shared<pulsar::KickoffMutator>(config.env),
      });

  std::vector<pulsar::TransitionEnginePtr> engines;
  engines.reserve(static_cast<std::size_t>(config.ppo.num_envs));
  for (int env_idx = 0; env_idx < config.ppo.num_envs; ++env_idx) {
    pulsar::EnvConfig env_config = config.env;
    env_config.seed += static_cast<std::uint64_t>(env_idx);
    engines.push_back(std::make_shared<pulsar::RocketSimTransitionEngine>(env_config, reset_mutator));
  }

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser = std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto reward_fn = std::make_shared<pulsar::CombinedRewardFunction>(config.reward);
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);

  const int updates = argc > 3 ? std::stoi(argv[3]) : 100;
  pulsar::PPOTrainer trainer(
      config,
      std::move(engines),
      obs_builder,
      action_parser,
      reward_fn,
      done_condition);
  trainer.train(updates, argv[2], argv[1]);
  return 0;
}
