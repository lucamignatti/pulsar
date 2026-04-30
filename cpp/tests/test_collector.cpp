#include <cstdlib>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "test_utils.hpp"

namespace {

class FakeTransitionEngine final : public pulsar::TransitionEngine {
 public:
  explicit FakeTransitionEngine(pulsar::EnvConfig config) : config_(std::move(config)) {
    reset(config_.seed);
  }

  void reset(std::uint64_t seed) override {
    pulsar::FixedTeamSizeMutator fixed(config_);
    pulsar::KickoffMutator kickoff(config_);
    state_ = {};
    fixed.apply(state_, seed);
    kickoff.apply(state_, seed);
    state_.goal_scored = false;
    state_.last_touch_tick = 0;
    steps_ = 0;
    for (std::size_t i = 0; i < state_.cars.size(); ++i) {
      state_.cars[i].boost = 10.0F + static_cast<float>(i);
      state_.cars[i].has_flip = true;
      state_.cars[i].on_ground = true;
    }
  }

  pulsar::StepResult step(std::span<const pulsar::ControllerState> actions) override {
    step_inplace(actions);
    return {.state = state_};
  }

  void step_inplace(std::span<const pulsar::ControllerState> actions) override {
    state_.tick += config_.tick_skip;
    state_.goal_scored = false;
    ++steps_;
    for (std::size_t i = 0; i < state_.cars.size(); ++i) {
      auto& car = state_.cars[i];
      car.velocity = {actions[i].steer * 100.0F, actions[i].throttle * 200.0F, 0.0F};
      car.position = car.position + car.velocity * 0.016F;
      car.is_boosting = actions[i].boost;
      car.boost = std::max(0.0F, car.boost - (actions[i].boost ? 1.0F : 0.0F));
      car.ball_touched = i == 0;
    }
    state_.last_touch_agent = 0;
    state_.last_touch_tick = state_.tick;
    if (steps_ >= 2) {
      state_.goal_scored = true;
      state_.blue_score += 1;
      state_.last_scoring_team = pulsar::Team::Blue;
      steps_ = 0;
    }
  }

  const pulsar::EnvState& state() const override { return state_; }
  std::size_t num_agents() const override { return state_.cars.size(); }

 private:
  pulsar::EnvConfig config_{};
  pulsar::EnvState state_{};
  int steps_ = 0;
};

void test_collector_shapes_self_play_and_reset() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.lfpo.num_envs = 2;
  config.lfpo.collection_workers = 0;
  config.env.max_episode_ticks = config.env.tick_skip;

  std::vector<pulsar::TransitionEnginePtr> engines;
  engines.push_back(std::make_shared<FakeTransitionEngine>(config.env));
  engines.push_back(std::make_shared<FakeTransitionEngine>(config.env));

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  pulsar::BatchedRocketSimCollector collector(
      config,
      std::move(engines),
      obs_builder,
      action_parser,
      done_condition,
      false);

  int assignment_calls = 0;
  collector.set_self_play_assignment_fn([&](std::size_t env_idx, std::uint64_t) {
    ++assignment_calls;
    return pulsar::SelfPlayAssignment{
        .enabled = true,
        .learner_team = env_idx == 0 ? pulsar::Team::Blue : pulsar::Team::Orange,
        .snapshot_index = static_cast<int>(7 + env_idx),
    };
  });

  const auto obs = collector.host_observations();
  pulsar::test::require(obs.sizes() == torch::IntArrayRef({8, 132}), "collector obs shape mismatch");
  pulsar::test::require(
      collector.host_action_masks().sizes() == torch::IntArrayRef({8, 90}),
      "collector action mask shape mismatch");
  pulsar::test::require(
      collector.host_learner_active().slice(0, 0, 4).sum().item<float>() == 2.0F,
      "env0 learner_active should map to blue team");
  pulsar::test::require(
      collector.host_learner_active().slice(0, 4, 8).sum().item<float>() == 2.0F,
      "env1 learner_active should map to orange team");
  pulsar::test::require(
      collector.host_snapshot_ids()[2].item<std::int64_t>() == 7,
      "env0 opponent snapshot id mismatch");
  pulsar::test::require(
      collector.host_snapshot_ids()[4].item<std::int64_t>() == 8,
      "env1 opponent snapshot id mismatch");

  std::vector<pulsar::ControllerState> actions(collector.total_agents(), {.throttle = 1.0F});
  collector.step(actions);
  pulsar::test::require(
      collector.host_observations().sizes() == torch::IntArrayRef({8, 132}),
      "collector next-step obs shape mismatch");
  pulsar::test::require(
      collector.host_dones().sum().item<float>() == 8.0F,
      "collector should report done after timeout/reset");
  pulsar::test::require(
      collector.host_episode_starts().sum().item<float>() == 8.0F,
      "episode_starts should mirror previous dones");
  pulsar::test::require(
      (collector.host_terminal_outcome_labels() == 2).all().item<bool>(),
      "timeout reset should produce neutral outcome labels");
  pulsar::test::require(assignment_calls >= 4, "assignments should be refreshed after reset");
}

void test_collector_goal_outcomes() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.lfpo.num_envs = 1;
  config.lfpo.collection_workers = 0;
  config.env.max_episode_ticks = config.env.tick_skip * 4;

  std::vector<pulsar::TransitionEnginePtr> engines;
  engines.push_back(std::make_shared<FakeTransitionEngine>(config.env));

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  pulsar::BatchedRocketSimCollector collector(
      config,
      std::move(engines),
      obs_builder,
      action_parser,
      done_condition,
      false);

  std::vector<pulsar::ControllerState> actions(collector.total_agents(), {.throttle = 1.0F});
  collector.step(actions);
  pulsar::test::require(
      (collector.host_terminal_outcome_labels() == 2).all().item<bool>(),
      "non-terminal transition should carry neutral outcome labels");
  collector.step(actions);
  pulsar::test::require(
      collector.host_terminal_outcome_labels().slice(0, 0, 2).sum().item<std::int64_t>() == 0,
      "blue scorers should receive score labels");
  pulsar::test::require(
      collector.host_terminal_outcome_labels().slice(0, 2, 4).sum().item<std::int64_t>() == 2,
      "orange conceders should receive concede labels");
}

void test_collector_parity_with_legacy_engine() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.lfpo.num_envs = 1;
  config.lfpo.collection_workers = 0;
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();

  auto reset_mutator = std::make_shared<pulsar::MutatorSequence>(
      std::vector<pulsar::StateMutatorPtr>{
          std::make_shared<pulsar::FixedTeamSizeMutator>(config.env),
          std::make_shared<pulsar::KickoffMutator>(config.env),
      });
  auto collector_engine = std::make_shared<pulsar::RocketSimTransitionEngine>(config.env, reset_mutator);
  collector_engine->reset(config.env.seed);

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  pulsar::BatchedRocketSimCollector collector(
      config,
      std::vector<pulsar::TransitionEnginePtr>{collector_engine},
      obs_builder,
      action_parser,
      done_condition,
      false);

  std::vector<float> oracle_obs(collector_engine->num_agents() * obs_builder->obs_dim());
  obs_builder->build_obs_batch(collector_engine->state(), oracle_obs);
  const auto collector_obs = collector.host_observations();
  const torch::Tensor oracle_obs_tensor =
      torch::from_blob(
          oracle_obs.data(),
          {static_cast<long>(collector_engine->num_agents()), static_cast<long>(obs_builder->obs_dim())},
          torch::kFloat32)
          .clone();
  pulsar::test::require(
      torch::allclose(collector_obs, oracle_obs_tensor, 1.0e-5, 1.0e-5),
      "collector obs parity mismatch");

  std::vector<pulsar::ControllerState> actions(collector_engine->num_agents(), {.throttle = 1.0F, .steer = 0.25F});
  collector.step(actions);
  pulsar::test::require(collector.host_dones().sum().item<float>() == 0.0F, "one-step parity run should not reset");
}

}  // namespace

int main() {
  try {
    test_collector_shapes_self_play_and_reset();
    test_collector_goal_outcomes();
    test_collector_parity_with_legacy_engine();
    std::cout << "pulsar_collector_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_collector_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
