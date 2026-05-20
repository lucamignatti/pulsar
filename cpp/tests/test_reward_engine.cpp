#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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
  cfg.dense_rewards.ball_touch_vel_weight = 5.0f;
  cfg.dense_rewards.speed_toward_ball_weight = 2.0f;
  cfg.dense_rewards.speed_toward_ball_decay = 300.0f;
  cfg.dense_rewards.face_ball_weight = 1.0f;
  cfg.dense_rewards.air_reward_weight = 0.15f;
  cfg.dense_rewards.air_reward_ball_z_min = 200.0f;
  cfg.dense_rewards.velocity_ball_to_goal_weight = 3.0f;
  cfg.dense_rewards.max_ball_speed = 6000.0f;
  cfg.dense_rewards.air_touch_weight = 2.0f;
  cfg.dense_rewards.air_touch_max_air_time = 1.75f;
  cfg.dense_rewards.save_boost_weight = 1.0f;
  cfg.dense_rewards.boost_pickup_big_weight = 3.0f;
  cfg.dense_rewards.boost_pickup_small_weight = 10.0f;
  cfg.dense_rewards.boost_pickup_big_threshold = 0.5f;
  cfg.mechanic_rewards.speed_flip = 0.5f;
  cfg.mechanic_rewards.wavedash = 0.3f;
  cfg.mechanic_rewards.chain_dash_bonus = 0.4f;
  cfg.mechanic_rewards.half_flip = 0.6f;
  cfg.mechanic_rewards.wall_dash = 0.35f;
  cfg.mechanic_rewards.air_dribble_base = 0.25f;
  cfg.mechanic_rewards.air_dribble_scale = 0.05f;
  cfg.mechanic_rewards.flip_reset = 0.7f;
  cfg.mechanic_rewards.ceiling_shot = 0.8f;
  cfg.mechanic_rewards.double_tap = 0.9f;
  cfg.mechanic_rewards.preflip = 0.45f;
  cfg.mechanic_rewards.redirect = 0.55f;
  cfg.mechanic_rewards.pogo = 0.65f;
  cfg.mechanic_rewards.pinch = 0.75f;
  cfg.mechanic_rewards.team_pinch = 0.85f;
  cfg.mechanic_rewards.kickoff_first_touch = 0.95f;
  return cfg;
}

void enable_possession_test_rewards(pulsar::ExperimentConfig& cfg) {
  cfg.dense_rewards.possession_chain_weight = 1.0f;
  cfg.dense_rewards.possession_chain_scale = 0.5f;
  cfg.dense_rewards.possession_chain_timeout_ticks = 360;
}

// Returns a neutral car state that produces zero gameplay rewards when the ball
// is at the origin and boost is zero.
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

    // =========================================================================
    // 5. Terminal reward - invalid label
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, true, 99);
      require_close(bd.terminal, 0.0f, "terminal invalid label");
    }

    // =========================================================================
    // 6. ball_touch_vel - touching ball with velocity delta
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.velocity = {500.0f, 0.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float delta_mag = 500.0f;
      float frac = 500.0f / 6000.0f;
      float expected = frac * 5.0f;
      require_close(bd.terms.at("gameplay.ball_touch_vel"), expected, "btv term");
      require_close(bd.gameplay, expected, "btv gameplay total");
    }

    // =========================================================================
    // 7. ball_touch_vel - NOT touching ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = false;
      pulsar::EnvState env{};
      env.ball.velocity = {500.0f, 0.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.ball_touch_vel"), 0.0f, "btv not touching");
    }

    // =========================================================================
    // 8. speed_toward_ball - moving toward ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.velocity = {0.0f, 300.0f, 0.0f};
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 300.0f, 17.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float decay = std::exp(-300.0f / 300.0f);
      float expected = 1.0f * decay * 2.0f;
      require_close(bd.terms.at("gameplay.speed_toward_ball"), expected, "stb toward");
      require_close(bd.gameplay, expected, "stb gameplay total");
    }

    // =========================================================================
    // 9. speed_toward_ball - moving away from ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.velocity = {0.0f, -300.0f, 0.0f};
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 300.0f, 17.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.speed_toward_ball"), 0.0f, "stb away");
    }

    // =========================================================================
    // 10. face_ball - facing the ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.forward = {0.0f, 1.0f, 0.0f};
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 300.0f, 17.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.face_ball"), 0.367879f, "face_ball facing");
    }

    // =========================================================================
    // 11. face_ball - facing away from ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.forward = {0.0f, -1.0f, 0.0f};
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 300.0f, 17.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.face_ball"), 0.0f, "face_ball away");
    }

    // =========================================================================
    // 12. air_reward - ball high and car airborne
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 300.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.air"), 0.15f, "air_reward airborne high ball");
    }

    // =========================================================================
    // 13. air_reward - ball position should not matter
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 100.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.air"), 0.15f, "air_reward ignores ball height");
    }

    // =========================================================================
    // 14. air_reward - car on ground
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 300.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.air"), 0.0f, "air_reward on ground");
    }

    // =========================================================================
    // 15. velocity_ball_to_goal - blue team, ball toward orange goal
    //     last_touch_agent must match car.id and be within the window.
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);  // car.id = 0
      pulsar::EnvState env{};
      env.ball.velocity = {0.0f, 3000.0f, 0.0f};
      env.last_touch_agent = car.id;   // this agent is credited
      env.last_touch_tick  = 0;        // touched this tick
      env.tick             = 0;
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float expected = (3000.0f / 6000.0f) * 3.0f;
      require_close(bd.terms.at("gameplay.velocity_ball_to_goal"), expected, "vbtg blue toward");
    }

    // =========================================================================
    // 16. velocity_ball_to_goal - orange team, ball toward their goal
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Orange);  // car.id = 0
      pulsar::EnvState env{};
      env.ball.velocity = {0.0f, -3000.0f, 0.0f};
      env.last_touch_agent = car.id;
      env.last_touch_tick  = 0;
      env.tick             = 0;
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float expected = (3000.0f / 6000.0f) * 3.0f;
      require_close(bd.terms.at("gameplay.velocity_ball_to_goal"), expected, "vbtg orange toward");
    }

    // =========================================================================
    // 17. velocity_ball_to_goal - ball moving the wrong way (negative reward)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      env.ball.velocity    = {0.0f, -3000.0f, 0.0f};  // toward own goal
      env.last_touch_agent = car.id;  // gate satisfied — direction must reject
      env.last_touch_tick  = 0;
      env.tick             = 0;
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.velocity_ball_to_goal"), -1.5f, "vbtg wrong way");
    }

    // =========================================================================
    // 18. air_touch - airborne + touching ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      car.ball_touched = true;
      car.air_time_since_jump = 1.0f;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 1022.0f};
      env.ball.velocity = {300.0f, 0.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float vel_delta_frac = 300.0f / 6000.0f;
      float height_frac = 1022.0f / 2044.0f;
      float air_time_frac = 1.0f / 1.75f;
      float quality = std::min(height_frac, air_time_frac);
      float expected = quality * vel_delta_frac * 2.0f;
      require_close(bd.terms.at("gameplay.air_touch"), expected, "air_touch term");
    }

    // =========================================================================
    // 19. air_touch - on ground
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 1022.0f};
      env.ball.velocity = {300.0f, 0.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.air_touch"), 0.0f, "air_touch on ground");
    }

    // =========================================================================
    // 20. save_boost - boost at 0.64 (sqrt = 0.8)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 0.64f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.64f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.save_boost"), 0.8f, "save_boost");
    }

    // =========================================================================
    // 21. save_boost - boost at zero
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 0.0f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.save_boost"), 0.0f, "save_boost zero");
    }

    // =========================================================================
    // 22. boost_pickup - big pad (delta > threshold)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 0.9f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.3f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.boost_pickup"), 0.6f * 3.0f, "boost big pad");
    }

    // =========================================================================
    // 23. boost_pickup - small pad (delta <= threshold)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 0.3f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.boost_pickup"), 0.3f * 10.0f, "boost small pad");
    }

    // =========================================================================
    // 24. boost_pickup - no change
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 0.5f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.5f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.boost_pickup"), 0.0f, "boost no change");
    }

    // =========================================================================
    // 24a. touch_direction - hit ball toward opponent goal
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.dense_rewards.touch_direction_weight = 20.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.position = {100.0f, 0.0f, 17.0f};
      env.ball.velocity = {500.0f, 800.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float vel_mag = std::sqrt(500.0f * 500.0f + 800.0f * 800.0f);
      float toward_goal = 800.0f / vel_mag;  // Blue goal is +y
      float delta_frac = vel_mag / 6000.0f;
      float expected = toward_goal * delta_frac * 20.0f;
      require_close(bd.terms.at("gameplay.touch_direction"), expected, "touch_direction toward goal");
    }

    // =========================================================================
    // 24b. touch_direction - hit toward own goal (should be zero)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.dense_rewards.touch_direction_weight = 20.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.velocity = {0.0f, -800.0f, 0.0f};  // toward Blue own goal (-y)
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.touch_direction"), 0.0f, "touch_direction zero when own goal");
    }

    // =========================================================================
    // 24c. boost_efficiency - engaged (near ball, full boost)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.dense_rewards.boost_efficiency_weight = 2.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 1.0f;  // 100 boost
      car.velocity = {1500.0f, 0.0f, 0.0f};  // high speed → engaged
      pulsar::EnvState env{};
      env.ball.position = {50.0f, 0.0f, 17.0f};  // ball nearby
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float expected = std::sqrt(1.0f) * 2.0f;  // sqrt(boost) * weight
      require_close(bd.terms.at("gameplay.boost_efficiency"), expected, "boost_efficiency engaged");
    }

    // =========================================================================
    // 24d. boost_efficiency - idle hoarding (far, slow, full boost → penalty)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.dense_rewards.boost_efficiency_weight = 2.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 0.6f;  // 60 boost (> 50 threshold)
      car.velocity = {100.0f, 0.0f, 0.0f};  // slow (< 200)
      pulsar::EnvState env{};
      env.ball.position = {5000.0f, 0.0f, 17.0f};  // ball very far (> 2000)
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float boost_amt = std::sqrt(0.6f);
      float expected = -boost_amt * 2.0f * 0.1f;  // penalty
      require_close(bd.terms.at("gameplay.boost_efficiency"), expected, "boost_efficiency idle penalty");
    }

    // =========================================================================
    // 24e. boost_used - boosting toward goal
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.dense_rewards.boost_used_weight = 2.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.is_boosting = true;
      car.boost = 0.33f;
      car.velocity = {0.0f, 1000.0f, 0.0f};  // toward Blue goal (+y)
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.66f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float boost_delta = 0.66f - 0.33f;
      float toward_goal_frac = 1.0f;  // velocity is all +y, Blue goal is +y
      require(bd.terms.at("gameplay.boost_used") > 0.0f, "boost_used positive");
    }

    // =========================================================================
    // 24f. defensive_positioning - between ball and own goal
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.dense_rewards.defensive_positioning_weight = 3.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.position = {0.0f, 0.0f, 17.0f};
      car.forward = {0.0f, -1.0f, 0.0f};  // facing own goal (-y for Blue)
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 500.0f, 17.0f};
      // opponent closer to ball
      pulsar::CarState opponent;
      opponent.id = 1;
      opponent.team = pulsar::Team::Orange;
      opponent.position = {0.0f, 400.0f, 17.0f};
      env.cars.push_back(opponent);
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      // car is between ball and own goal, opponent has ball → positive reward
      require(bd.terms.at("gameplay.defensive_positioning") > 0.0f, "defensive_positioning positive");
    }

    // =========================================================================
    // 24g. shot_accuracy - ball heading toward goal from offensive half after touch
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.dense_rewards.shot_accuracy_weight = 5.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.position = {0.0f, 500.0f, 17.0f};  // in offensive half (+y)
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.velocity = {0.0f, 3000.0f, 0.0f};  // toward Blue goal (+y), fast
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      float toward_goal = 1.0f;
      float speed_frac = 3000.0f / 6000.0f;
      float expected = toward_goal * speed_frac * 5.0f;
      require_close(bd.terms.at("gameplay.shot_accuracy"), expected, "shot_accuracy positive");
    }

    // =========================================================================
    // 25. speed_flip - was on ground, now flipping near ground, boosting
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      car.position = {0.0f, 0.0f, 50.0f};  // near ground (kGroundZ=17, threshold 150)
      car.is_flipping = true;
      car.is_boosting = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = true;
      agent_state.prev_is_flipping = false;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.speed_flip"), 0.5f, "speed_flip");
    }

    // =========================================================================
    // 26. speed_flip - not flipping (should not fire)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.is_flipping = false;
      car.is_boosting = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_is_flipping = false;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.speed_flip"), 0.0f, "speed_flip no flip");
    }

    // =========================================================================
    // 27. wavedash - landing after flipping near ground
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;  // just landed
      car.position = {0.0f, 0.0f, 17.0f};
      car.is_flipping = false;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;  // was airborne
      agent_state.prev_is_flipping = true;  // was flipping
      agent_state.prev_car_position_z = 37.0f;  // was near ground (kGroundZ=17, threshold=70)
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(100, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.wavedash"), 0.3f, "wavedash");
      require(agent_state.last_wavedash_tick == 100, "wavedash sets last_wavedash_tick");
    }

    // =========================================================================
    // 28. wavedash - too high above ground (should not fire)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;  // landed
      car.position = {0.0f, 0.0f, 17.0f};
      car.is_flipping = false;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;
      agent_state.prev_is_flipping = true;
      agent_state.prev_car_position_z = 120.0f;  // too high (above kWavedashZThreshold=70)
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.wavedash"), 0.0f, "wavedash too high");
    }

    // =========================================================================
    // 29. chain_dash - wavedash within window
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;  // just landed
      car.position = {0.0f, 0.0f, 17.0f};
      car.is_flipping = false;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;
      agent_state.prev_is_flipping = true;
      agent_state.prev_car_position_z = 37.0f;
      agent_state.last_wavedash_tick = 90;  // previous wavedash at tick 90
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(110, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.wavedash"), 0.3f, "chain_dash wavedash component");
      require_close(bd.terms.at("mechanic.chain_dash"), 0.4f, "chain_dash");
    }

    // =========================================================================
    // 30. chain_dash - wavedash outside window
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;  // just landed
      car.position = {0.0f, 0.0f, 17.0f};
      car.is_flipping = false;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;
      agent_state.prev_is_flipping = true;
      agent_state.prev_car_position_z = 37.0f;
      agent_state.last_wavedash_tick = 50;  // too far back (kChainDashWindowTicks=30)
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(110, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.chain_dash"), 0.0f, "chain_dash outside window");
    }

    // =========================================================================
    // 31. half_flip - backward velocity, handbrake, airborne flip
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      car.is_flipping = true;
      car.has_double_jumped = false;
      car.velocity = {-500.0f, 0.0f, 0.0f};
      car.forward = {1.0f, 0.0f, 0.0f};
      car.handbrake = 0.8f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_is_flipping = false;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.half_flip"), 0.6f, "half_flip");
    }

    // =========================================================================
    // 32. half_flip - on ground (should not fire)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.is_flipping = true;
      car.velocity = {-500.0f, 0.0f, 0.0f};
      car.handbrake = 0.8f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_is_flipping = false;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.half_flip"), 0.0f, "half_flip on ground");
    }

    // =========================================================================
    // 33. wall_dash - landing near wall after flipping near ground
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.position = {4050.0f, 0.0f, 17.0f};
      car.is_flipping = false;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;
      agent_state.prev_is_flipping = true;
      agent_state.prev_car_position_z = 37.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.wall_dash"), 0.35f, "wall_dash near x-wall");
    }

    // =========================================================================
    // 34. wall_dash - not near wall
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.position = {0.0f, 0.0f, 17.0f};
      car.is_flipping = false;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;
      agent_state.prev_is_flipping = true;
      agent_state.prev_car_position_z = 37.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.wall_dash"), 0.0f, "wall_dash not near wall");
    }

    // =========================================================================
    // 35. air_dribble - first airborne touch
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 500.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.consecutive_air_touches = 0;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.consecutive_air_touches == 1, "air_dribble increments counter");
      require_close(bd.terms.at("mechanic.air_dribble"), 0.25f, "air_dribble first touch");
    }

    // =========================================================================
    // 36. air_dribble - second consecutive airborne touch
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 500.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.consecutive_air_touches = 1;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.air_dribble"), 0.25f + 0.05f, "air_dribble second touch");
    }

    // =========================================================================
    // 37. air_dribble - reset on ground
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.ball_touched = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.consecutive_air_touches = 3;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.consecutive_air_touches == 0, "air_dribble resets on ground");
      require_close(bd.terms.at("mechanic.air_dribble"), 0.0f, "air_dribble reset");
    }

    // =========================================================================
    // 38. flip_reset - transition to has_flip_reset
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.has_flip_reset = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_has_flip_reset = false;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.flip_reset"), 0.7f, "flip_reset");
    }

    // =========================================================================
    // 39. flip_reset - already had it (should not fire)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.has_flip_reset = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_has_flip_reset = true;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.flip_reset"), 0.0f, "flip_reset already had");
    }

    // =========================================================================
    // 40. ceiling_shot - go to ceiling then touch ball while falling
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      car.position.z = 1900.0f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      pulsar::EnvRewardState env_state{};
      engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.was_on_ceiling, "was_on_ceiling set");

      auto car2 = make_neutral_car(pulsar::Team::Blue);
      car2.on_ground = false;
      car2.position.z = 1600.0f;
      car2.ball_touched = true;
      pulsar::AgentRewardState agent_state2{};
      agent_state2.was_on_ceiling = true;
      agent_state2.prev_ball_touched = false;
      auto bd = engine.compute(1, car2, env, 2, agent_state2, env_state, false, 0);
      require_close(bd.terms.at("mechanic.ceiling_shot"), 0.8f, "ceiling_shot");
      require(!agent_state2.was_on_ceiling, "was_on_ceiling cleared");
    }

    // =========================================================================
    // 41. double_tap - state tracking (ball_hit_backboard, last_toucher)
    //     The reward check runs BEFORE last_ball_touch_tick is updated, so
    //     the tick delta is correctly > 0 on subsequent touches.
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.id = 1;
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.position.y = 5200.0f;
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      pulsar::EnvRewardState env_state{};
      engine.compute(100, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.ball_hit_backboard, "ball_hit_backboard set");
      require(agent_state.last_toucher_id == 1, "last_toucher_id set");

      auto car2 = make_neutral_car(pulsar::Team::Blue);
      car2.id = 1;
      car2.ball_touched = true;
      env.ball.position.y = 4000.0f;
      pulsar::AgentRewardState agent_state2{};
      agent_state2.ball_hit_backboard = true;
      agent_state2.prev_ball_touched = false;
      agent_state2.last_ball_touch_tick = 100;
      agent_state2.last_toucher_id = 1;
      auto bd = engine.compute(110, car2, env, 2, agent_state2, env_state, false, 0);
      require(agent_state2.last_ball_touch_tick == 110, "last_ball_touch_tick updated");
      require(agent_state2.last_toucher_id == 1, "last_toucher_id preserved");
    }

    // =========================================================================
    // 42. preflip - start flip then touch ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.is_flipping = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_is_flipping = false;
      pulsar::EnvRewardState env_state{};
      engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.flip_start_tick == 1, "flip_start_tick set");

      auto car2 = make_neutral_car(pulsar::Team::Blue);
      car2.is_flipping = true;
      car2.ball_touched = true;
      pulsar::AgentRewardState agent_state2{};
      agent_state2.prev_is_flipping = true;
      agent_state2.prev_ball_touched = false;
      agent_state2.flip_start_tick = 1;
      auto bd = engine.compute(1, car2, env, 2, agent_state2, env_state, false, 0);
      require_close(bd.terms.at("mechanic.preflip"), 0.45f, "preflip");
    }

    // =========================================================================
    // 43. redirect - ball changes direction toward goal
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.velocity = {0.0f, 500.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {300.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.redirect"), 0.55f, "redirect");
    }

    // =========================================================================
    // 44. redirect - not enough prev speed
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.ball.velocity = {0.0f, 500.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {50.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.redirect"), 0.0f, "redirect low prev speed");
    }

    // =========================================================================
    // 45. pogo - land on ground with upward velocity, no jump
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.is_jumping = false;
      car.has_jumped = false;
      car.velocity.z = 300.0f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;
      // pogo checks the PRE-LANDING velocity (saved from the previous tick)
      agent_state.prev_car_velocity_z = 300.0f;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.pogo"), 0.65f, "pogo");
    }

    // =========================================================================
    // 46. pogo - downward velocity (should not fire)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = true;
      car.is_jumping = false;
      car.has_jumped = false;
      car.velocity.z = -100.0f;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = false;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.pogo"), 0.0f, "pogo downward");
    }

    // =========================================================================
    // 47. pinch - ball velocity increases by > kPinchVelocityDelta
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      // position ball directly in front of car at same height
      pulsar::EnvState env{};
      env.ball.position = {100.0f, 0.0f, 17.0f};
      env.ball.velocity = {1000.0f, 0.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.pinch"), 0.75f, "pinch");
    }

    // =========================================================================
    // 48. pinch - velocity increase below threshold
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      env.ball.velocity = {500.0f, 0.0f, 0.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {300.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.pinch"), 0.0f, "pinch below threshold");
    }

    // =========================================================================
    // 49. team_pinch - pinch with teammate near ball
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      pulsar::CarState car = make_neutral_car(pulsar::Team::Blue);
      car.id = 0;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 0.0f, 100.0f};
      env.ball.velocity = {0.0f, 0.0f, 1000.0f};
      pulsar::CarState teammate;
      teammate.id = 1;
      teammate.team = pulsar::Team::Blue;
      teammate.position = {50.0f, 0.0f, 100.0f};
      env.cars.push_back(teammate);
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.team_pinch"), 0.85f, "team_pinch");
    }

    // =========================================================================
    // 50. team_pinch - pinch but teammate too far
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      pulsar::CarState car = make_neutral_car(pulsar::Team::Blue);
      car.id = 0;
      pulsar::EnvState env{};
      env.ball.velocity = {1000.0f, 0.0f, 0.0f};
      env.ball.position = {0.0f, 0.0f, 100.0f};
      pulsar::CarState teammate;
      teammate.id = 1;
      teammate.team = pulsar::Team::Blue;
      teammate.position = {2000.0f, 0.0f, 100.0f};
      env.cars.push_back(teammate);
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.team_pinch"), 0.0f, "team_pinch too far");
    }

    // =========================================================================
    // 51. kickoff_first_touch - first to touch (no opponent)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.kickoff_first_touch"), 0.95f, "kickoff_first_touch");
      require(env_state.kickoff_reward_given, "kickoff_reward_given set");
      require(env_state.first_touch_team == 0, "first_touch_team blue");
    }

    // =========================================================================
    // 52. kickoff_first_touch - already given (should not fire again)
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      env_state.kickoff_reward_given = true;
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.kickoff_first_touch"), 0.0f, "kickoff already given");
    }

    // =========================================================================
    // 53. kickoff_50_50_touch - both teams touch at kickoff
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.mechanic_rewards.kickoff_50_50_touch = 0.95f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.id = 0;
      car.ball_touched = true;
      pulsar::EnvState env{};
      pulsar::CarState opponent;
      opponent.id = 1;
      opponent.team = pulsar::Team::Orange;
      opponent.ball_touched = true;
      env.cars.push_back(opponent);
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("mechanic.kickoff_first_touch"), 0.0f, "kickoff contested first touch");
      require_close(bd.terms.at("mechanic.kickoff_50_50_touch"), 0.95f, "kickoff contested 50/50");
      require(env_state.first_touch_team == 2, "first_touch_team contested");
      require(env_state.kickoff_reward_given, "kickoff_reward_given set on 50/50");
    }

    // =========================================================================
    // 54. State bookkeeping - prev_boost and prev_ball_velocity updated
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.boost = 0.42f;
      pulsar::EnvState env{};
      env.ball.velocity = {100.0f, 200.0f, 300.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_boost = 0.0f;
      agent_state.prev_ball_velocity = {0.0f, 0.0f, 0.0f};
      pulsar::EnvRewardState env_state{};
      engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require_close(agent_state.prev_boost, 0.42f, "prev_boost updated");
      require_close(agent_state.prev_ball_velocity.x, 100.0f, "prev_ball_vel.x updated");
      require_close(agent_state.prev_ball_velocity.y, 200.0f, "prev_ball_vel.y updated");
      require_close(agent_state.prev_ball_velocity.z, 300.0f, "prev_ball_vel.z updated");
    }

    // =========================================================================
    // 55. State bookkeeping - on_ground, has_flip, etc. updated
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.on_ground = false;
      car.has_flip = false;
      car.has_flipped = true;
      car.is_flipping = true;
      car.has_flip_reset = true;
      car.ball_touched = true;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.prev_on_ground == false, "prev_on_ground");
      require(agent_state.prev_has_flip == false, "prev_has_flip");
      require(agent_state.prev_has_flipped == true, "prev_has_flipped");
      require(agent_state.prev_is_flipping == true, "prev_is_flipping");
      require(agent_state.prev_has_flip_reset == true, "prev_has_flip_reset");
      require(agent_state.prev_ball_touched == true, "prev_ball_touched");
    }

    // =========================================================================
    // 56. Breakdown terms map has all expected keys
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, false, 0);
      require(bd.terms.find("gameplay.ball_touch_vel") != bd.terms.end(), "terms has ball_touch_vel");
      require(bd.terms.find("gameplay.speed_toward_ball") != bd.terms.end(), "terms has speed_toward_ball");
      require(bd.terms.find("gameplay.face_ball") != bd.terms.end(), "terms has face_ball");
      require(bd.terms.find("gameplay.air") != bd.terms.end(), "terms has air");
      require(bd.terms.find("gameplay.velocity_ball_to_goal") != bd.terms.end(), "terms has vbtg");
      require(bd.terms.find("gameplay.air_touch") != bd.terms.end(), "terms has air_touch");
      require(bd.terms.find("gameplay.save_boost") != bd.terms.end(), "terms has save_boost");
      require(bd.terms.find("gameplay.boost_pickup") != bd.terms.end(), "terms has boost_pickup");
      require(bd.terms.find("gameplay.touch_direction") != bd.terms.end(), "terms has touch_direction");
      require(bd.terms.find("gameplay.boost_efficiency") != bd.terms.end(), "terms has boost_efficiency");
      require(bd.terms.find("gameplay.boost_used") != bd.terms.end(), "terms has boost_used");
      require(bd.terms.find("gameplay.defensive_positioning") != bd.terms.end(), "terms has defensive_positioning");
      require(bd.terms.find("gameplay.shot_accuracy") != bd.terms.end(), "terms has shot_accuracy");
      require(bd.terms.find("gameplay.possession_chain") != bd.terms.end(), "terms has possession_chain");
      require(bd.terms.find("mechanic.speed_flip") != bd.terms.end(), "terms has speed_flip");
      require(bd.terms.find("mechanic.wavedash") != bd.terms.end(), "terms has wavedash");
      require(bd.terms.find("mechanic.chain_dash") != bd.terms.end(), "terms has chain_dash");
      require(bd.terms.find("mechanic.half_flip") != bd.terms.end(), "terms has half_flip");
      require(bd.terms.find("mechanic.wall_dash") != bd.terms.end(), "terms has wall_dash");
      require(bd.terms.find("mechanic.air_dribble") != bd.terms.end(), "terms has air_dribble");
      require(bd.terms.find("mechanic.flip_reset") != bd.terms.end(), "terms has flip_reset");
      require(bd.terms.find("mechanic.ceiling_shot") != bd.terms.end(), "terms has ceiling_shot");
      require(bd.terms.find("mechanic.double_tap") != bd.terms.end(), "terms has double_tap");
      require(bd.terms.find("mechanic.preflip") != bd.terms.end(), "terms has preflip");
      require(bd.terms.find("mechanic.redirect") != bd.terms.end(), "terms has redirect");
      require(bd.terms.find("mechanic.pogo") != bd.terms.end(), "terms has pogo");
      require(bd.terms.find("mechanic.pinch") != bd.terms.end(), "terms has pinch");
      require(bd.terms.find("mechanic.team_pinch") != bd.terms.end(), "terms has team_pinch");
      require(bd.terms.find("mechanic.kickoff_first_touch") != bd.terms.end(), "terms has kickoff_first_touch");
      require(bd.terms.find("mechanic.kickoff_50_50_touch") != bd.terms.end(), "terms has kickoff_50_50_touch");
    }

    // =========================================================================
    // 57. Combined reward - gameplay + mechanic + terminal
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      cfg.mechanic_rewards.wavedash = 0.0f;
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.forward = {0.0f, 1.0f, 0.0f};
      car.on_ground = false;
      car.position = {0.0f, 0.0f, 50.0f};  // near ground for speed_flip
      car.is_flipping = true;
      car.is_boosting = true;
      pulsar::EnvState env{};
      env.ball.position = {0.0f, 300.0f, 17.0f};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_on_ground = true;
      agent_state.prev_is_flipping = false;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(0, car, env, 2, agent_state, env_state, true, 0);
      require_close(bd.gameplay, 0.513475f, "combined gameplay");
      require_close(bd.mechanic, 0.5f, "combined mechanic");
      require_close(bd.terminal, 10.0f, "combined terminal");
      require_close(bd.total, 11.013475f, "combined total");
    }

    // =========================================================================
    // 62. Possession chain - first touch gives base reward
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      enable_possession_test_rewards(cfg);
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.last_touch_agent = car.id;
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.prev_env_last_touch_agent = -1;
      agent_state.consecutive_touches = 0;
      agent_state.last_own_touch_tick = -1000;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(100, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.possession_chain"), 1.0f, "pc first touch base");
      require(agent_state.consecutive_touches == 1, "pc consecutive_touches = 1");
      require(agent_state.last_own_touch_tick == 100, "pc last_own_touch_tick = 100");
    }

    // =========================================================================
    // 63. Possession chain - second consecutive touch gets scale bonus
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      enable_possession_test_rewards(cfg);
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.last_touch_agent = car.id;
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.prev_env_last_touch_agent = car.id;  // same agent previously
      agent_state.consecutive_touches = 1;
      agent_state.last_own_touch_tick = 90;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(100, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.possession_chain"), 1.5f, "pc second touch base+scale");
      require(agent_state.consecutive_touches == 2, "pc consecutive_touches = 2");
    }

    // =========================================================================
    // 64. Possession chain - opponent touch resets chain
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      enable_possession_test_rewards(cfg);
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.last_touch_agent = car.id;  // current tick: this agent touched
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.prev_env_last_touch_agent = 1;  // previous tick: opponent touched
      agent_state.consecutive_touches = 2;
      agent_state.last_own_touch_tick = 80;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(100, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.consecutive_touches == 1, "pc opponent reset chain to 1");
      require_close(bd.terms.at("gameplay.possession_chain"), 1.0f, "pc after opponent reset");
    }

    // =========================================================================
    // 65. Possession chain - timeout resets chain
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      enable_possession_test_rewards(cfg);
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = true;
      pulsar::EnvState env{};
      env.last_touch_agent = car.id;
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.prev_env_last_touch_agent = car.id;
      agent_state.consecutive_touches = 5;
      agent_state.last_own_touch_tick = 0;  // 500 ticks ago, > 360 timeout
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(500, car, env, 2, agent_state, env_state, false, 0);
      require(agent_state.consecutive_touches == 1, "pc timeout reset chain to 1");
      require_close(bd.terms.at("gameplay.possession_chain"), 1.0f, "pc after timeout reset");
    }

    // =========================================================================
    // 66. Possession chain - no touch gives zero reward
    // =========================================================================
    {
      auto cfg = make_reward_test_config();
      enable_possession_test_rewards(cfg);
      pulsar::RewardEngine engine(cfg);
      auto car = make_neutral_car(pulsar::Team::Blue);
      car.ball_touched = false;
      pulsar::EnvState env{};
      pulsar::AgentRewardState agent_state{};
      agent_state.prev_ball_touched = false;
      agent_state.prev_env_last_touch_agent = -1;
      agent_state.consecutive_touches = 0;
      agent_state.last_own_touch_tick = -1000;
      pulsar::EnvRewardState env_state{};
      auto bd = engine.compute(100, car, env, 2, agent_state, env_state, false, 0);
      require_close(bd.terms.at("gameplay.possession_chain"), 0.0f, "pc no touch zero");
      require(agent_state.consecutive_touches == 0, "pc no change");
    }

    std::cout << "test_reward_engine passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "test_reward_engine FAILED: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
