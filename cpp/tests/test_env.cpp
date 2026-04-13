#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "test_utils.hpp"

namespace {

bool nearly_equal(float lhs, float rhs, float tol = 1.0e-5F) {
  return std::fabs(lhs - rhs) <= tol;
}

void test_obs_content_and_done() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();

  pulsar::EnvState state;
  pulsar::FixedTeamSizeMutator fixed(config.env);
  pulsar::KickoffMutator kickoff(config.env);
  fixed.apply(state, 11);
  kickoff.apply(state, 11);

  state.ball.position = {230.0F, -460.0F, 920.0F};
  state.ball.velocity = {115.0F, -230.0F, 345.0F};
  state.ball.angular_velocity = {3.14159265F, 0.0F, -3.14159265F};
  state.cars[0].position = {460.0F, 230.0F, 17.0F};
  state.cars[0].velocity = {230.0F, -460.0F, 0.0F};
  state.cars[0].forward = {0.0F, 1.0F, 0.0F};
  state.cars[0].up = {0.0F, 0.0F, 1.0F};
  state.cars[0].boost = 42.0F;
  state.cars[0].on_ground = false;
  state.cars[0].is_boosting = true;
  state.cars[0].has_flip = false;
  state.cars[0].air_time_since_jump = 0.75F;
  state.cars[1].position = {920.0F, 0.0F, 17.0F};
  state.cars[2].position = {-1150.0F, 575.0F, 17.0F};

  pulsar::PulsarObsBuilder obs_builder(config.env);
  const auto obs = obs_builder.build_obs(state, 0);
  pulsar::test::require(obs.size() == obs_builder.obs_dim(), "unexpected observation size");
  pulsar::test::require(nearly_equal(obs[0], state.ball.position.x / 2300.0F), "ball x scale mismatch");
  pulsar::test::require(nearly_equal(obs[1], state.ball.position.y / 2300.0F), "ball y scale mismatch");
  pulsar::test::require(nearly_equal(obs[9 + state.boost_pad_timers.size()], 0.0F), "holding jump flag mismatch");
  pulsar::test::require(nearly_equal(obs[67], state.cars[0].boost * 0.01F), "self boost slot mismatch");
  pulsar::test::require(nearly_equal(obs[69], 0.0F), "self on_ground slot mismatch");
  pulsar::test::require(nearly_equal(obs[70], 1.0F), "self is_boosting slot mismatch");
  pulsar::test::require(nearly_equal(obs[72], state.cars[1].position.x / 2300.0F), "ally ordering mismatch");
  pulsar::test::require(nearly_equal(obs[92], state.cars[2].position.x / 2300.0F), "enemy ordering mismatch");

  pulsar::SimpleDoneCondition done(config.env);
  pulsar::EnvState timeout_state = state;
  timeout_state.last_touch_tick = 0;
  timeout_state.tick =
      static_cast<int>(config.env.no_touch_timeout_seconds * static_cast<float>(config.env.tick_rate));
  const auto [terminated, truncated] = done.is_done(timeout_state, config.env.tick_skip);
  pulsar::test::require(terminated[0] == 0, "no-touch timeout should not terminate");
  pulsar::test::require(truncated[0] == 1, "no-touch timeout should truncate");

  timeout_state.goal_scored = true;
  timeout_state.tick = 0;
  timeout_state.last_touch_tick = 0;
  const auto [goal_terminated, goal_truncated] = done.is_done(timeout_state, 0);
  pulsar::test::require(goal_terminated[0] == 1, "goal should terminate");
  pulsar::test::require(goal_truncated[0] == 0, "goal alone should not truncate");

  timeout_state.goal_scored = false;
  const auto [tick_terminated, tick_truncated] = done.is_done(timeout_state, config.env.max_episode_ticks);
  pulsar::test::require(tick_terminated[0] == 0, "max tick timeout should not terminate");
  pulsar::test::require(tick_truncated[0] == 1, "max tick timeout should truncate");
}

void test_mutators() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  pulsar::EnvState state;
  pulsar::FixedTeamSizeMutator fixed(config.env);
  pulsar::KickoffMutator kickoff(config.env);
  fixed.apply(state, 5);
  kickoff.apply(state, 5);

  pulsar::test::require(state.cars.size() == 4, "2v2 mutator should create four cars");
  pulsar::test::require(state.cars[0].team == pulsar::Team::Blue, "first car should be blue");
  pulsar::test::require(state.cars[2].team == pulsar::Team::Orange, "third car should be orange");
  pulsar::test::require(std::fabs(state.ball.position.x) < 1.0e-5F, "kickoff ball x should be centered");
  pulsar::test::require(std::fabs(state.ball.position.y) < 1.0e-5F, "kickoff ball y should be centered");
}

void test_rocketsim_reproducibility() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  auto reset_mutator = std::make_shared<pulsar::MutatorSequence>(
      std::vector<pulsar::StateMutatorPtr>{
          std::make_shared<pulsar::FixedTeamSizeMutator>(config.env),
          std::make_shared<pulsar::KickoffMutator>(config.env),
      });

  pulsar::RocketSimTransitionEngine first(config.env, reset_mutator);
  pulsar::RocketSimTransitionEngine second(config.env, reset_mutator);
  first.reset(19);
  second.reset(19);

  const auto& first_state = first.state();
  const auto& second_state = second.state();
  pulsar::test::require(first_state.cars.size() == second_state.cars.size(), "reset agent count mismatch");
  pulsar::test::require(nearly_equal(first_state.ball.position.x, second_state.ball.position.x), "reset ball x mismatch");
  pulsar::test::require(nearly_equal(first_state.ball.position.y, second_state.ball.position.y), "reset ball y mismatch");
  pulsar::test::require(first_state.blue_score == second_state.blue_score, "reset score mismatch");

  std::vector<pulsar::ControllerState> actions(
      first.num_agents(),
      pulsar::ControllerState{.throttle = 1.0F, .steer = 0.25F, .boost = true});
  first.step_inplace(actions);
  second.step_inplace(actions);

  const auto& stepped_first = first.state();
  const auto& stepped_second = second.state();
  pulsar::test::require(stepped_first.cars.size() == stepped_second.cars.size(), "step agent count mismatch");
  pulsar::test::require(nearly_equal(stepped_first.ball.position.x, stepped_second.ball.position.x), "step ball x mismatch");
  pulsar::test::require(nearly_equal(stepped_first.cars[0].position.y, stepped_second.cars[0].position.y), "step car position mismatch");
  pulsar::test::require(stepped_first.cars[0].is_boosting == stepped_second.cars[0].is_boosting, "boost flag mismatch");
}

}  // namespace

int main() {
  try {
    test_obs_content_and_done();
    test_mutators();
    test_rocketsim_reproducibility();
    std::cout << "pulsar_env_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_env_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
