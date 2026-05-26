#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "pulsar/training/reward_engine.hpp"
#include "test_utils.hpp"

namespace {

const float kTol = 1.0e-4F;

void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

void require_close(float a, float b, const std::string& msg) {
  if (std::fabs(a - b) > kTol) {
    throw std::runtime_error(msg + " expected " + std::to_string(b) + " got " + std::to_string(a));
  }
}

pulsar::ExperimentConfig make_reward_test_config() {
  auto cfg = pulsar::test::make_test_config();
  cfg.outcome.score = 10.0f;
  cfg.outcome.concede = -8.0f;
  cfg.outcome.neutral = 0.0f;
  cfg.outcome.neutral_no_touch = -1.0f;
  return cfg;
}

// Returns a neutral car state
pulsar::CarState make_neutral_car(pulsar::Team team = pulsar::Team::Blue) {
  pulsar::CarState c;
  c.id = 0;
  c.team = team;
  c.position = {0.0f, 0.0f, 17.0f};
  c.velocity = {0.0f, 0.0f, 0.0f};
  c.forward = {1.0f, 0.0f, 0.0f};
  c.up = {0.0f, 0.0f, 1.0f};
  c.boost = 0.0f;
  c.on_ground = true;
  c.has_flip = true;
  c.has_flip_reset = false;
  c.ball_touched = false;
  c.is_flipping = false;
  c.is_boosting = false;
  c.has_double_jumped = false;
  c.has_flipped = false;
  c.handbrake = 0.0f;
  c.air_time_since_jump = 0.0f;
  return c;
}

}  // namespace

int main() {
  try {
    // =========================================================================
    // 1. Terminal reward - score
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, true, 0);
      require_close(bd.terminal, 10.0f, "terminal score");
      require_close(bd.total, 10.0f, "total score");
      auto it = bd.terms.find("terminal.score");
      require(it != bd.terms.end(), "terms has terminal.score");
      require_close(it->second, 10.0f, "terms terminal.score value");
    }

    // =========================================================================
    // 2. Terminal reward - concede
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, true, 1);
      require_close(bd.terminal, -8.0f, "terminal concede");
      require_close(bd.total, -8.0f, "total concede");
      auto it = bd.terms.find("terminal.concede");
      require(it != bd.terms.end(), "terms has terminal.concede");
      require_close(it->second, -8.0f, "terms terminal.concede value");
    }

    // =========================================================================
    // 3. Terminal reward - neutral
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, true, 2);
      require_close(bd.terminal, 0.0f, "terminal neutral");
      require_close(bd.total, 0.0f, "total neutral");
      auto it = bd.terms.find("terminal.neutral");
      require(it != bd.terms.end(), "terms has terminal.neutral");
      require_close(it->second, 0.0f, "terms terminal.neutral value");
    }

    // =========================================================================
    // 4. Terminal reward - no_touch
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, true, 3);
      require_close(bd.terminal, -1.0f, "terminal no_touch");
      require_close(bd.total, -1.0f, "total no_touch");
      auto it = bd.terms.find("terminal.no_touch");
      require(it != bd.terms.end(), "terms has terminal.no_touch");
      require_close(it->second, -1.0f, "terms terminal.no_touch value");
    }

    std::cout << "All reward engine tests passed successfully!\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
