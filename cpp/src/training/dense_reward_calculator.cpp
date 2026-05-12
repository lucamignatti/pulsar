#include "pulsar/training/dense_reward_calculator.hpp"

#include <algorithm>
#include <cmath>

namespace pulsar {

namespace {

float vec3_magnitude(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float vec3_dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float safe_division(float numerator, float denominator) {
  return numerator / std::max(denominator, 1.0e-6F);
}

float clamp(float value, float lo, float hi) {
  return std::min(std::max(value, lo), hi);
}

}  // namespace

DenseRewardCalculator::DenseRewardCalculator(const DenseRewardConfig& cfg) : cfg_(cfg) {}

void DenseRewardCalculator::update_config(const DenseRewardConfig& cfg) {
  cfg_ = cfg;
}

float DenseRewardCalculator::update(
    const CarState& car,
    const EnvState& env,
    AgentDenseState& s) const {
  float reward = 0.0F;

  const float vel_delta = ball_touch_vel(car, env, s);
  reward += vel_delta;
  reward += speed_toward_ball(car, env);
  reward += face_ball(car, env);
  reward += air_reward(car, env);
  reward += velocity_ball_to_goal(car, env);

  const float vel_delta_frac = safe_division(vel_delta, cfg_.ball_touch_vel_weight);
  reward += air_touch(car, env, vel_delta_frac);
  reward += save_boost(car);
  reward += boost_pickup(car, s);

  const float cap = cfg_.dense_reward_cap_per_episode;
  if (cap > 0.0F) {
    const float remaining = cap - s.episode_dense_reward;
    if (remaining <= 0.0F) {
      reward = 0.0F;
    } else if (reward > remaining) {
      reward = remaining;
    }
  }
  s.episode_dense_reward += reward;

  s.prev_boost = car.boost;
  s.prev_ball_velocity = env.ball.velocity;

  return reward;
}

float DenseRewardCalculator::ball_touch_vel(
    const CarState& car, const EnvState& env, AgentDenseState& s) const {
  if (cfg_.ball_touch_vel_weight <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;

  const Vec3 delta{
      env.ball.velocity.x - s.prev_ball_velocity.x,
      env.ball.velocity.y - s.prev_ball_velocity.y,
      env.ball.velocity.z - s.prev_ball_velocity.z,
  };
  const float delta_mag = vec3_magnitude(delta);
  return clamp(delta_mag / std::max(cfg_.max_ball_speed, 1.0F), 0.0F, 1.0F)
      * cfg_.ball_touch_vel_weight;
}

float DenseRewardCalculator::speed_toward_ball(
    const CarState& car, const EnvState& env) const {
  if (cfg_.speed_toward_ball_weight <= 0.0F) return 0.0F;

  const Vec3 to_ball{
      env.ball.position.x - car.position.x,
      env.ball.position.y - car.position.y,
      env.ball.position.z - car.position.z,
  };
  const float dist = vec3_magnitude(to_ball);
  if (dist < 1.0F) return 0.0F;

  const Vec3 to_ball_norm{
      to_ball.x / dist,
      to_ball.y / dist,
      to_ball.z / dist,
  };

  const float speed_toward = vec3_dot(car.velocity, to_ball_norm);
  const float car_speed = vec3_magnitude(car.velocity);
  const float frac = clamp(speed_toward / std::max(car_speed, 1.0F), 0.0F, 1.0F);
  const float decay = std::exp(-dist / cfg_.speed_toward_ball_decay);

  return frac * decay * cfg_.speed_toward_ball_weight;
}

float DenseRewardCalculator::face_ball(
    const CarState& car, const EnvState& env) const {
  if (cfg_.face_ball_weight <= 0.0F) return 0.0F;

  const Vec3 to_ball{
      env.ball.position.x - car.position.x,
      env.ball.position.y - car.position.y,
      env.ball.position.z - car.position.z,
  };
  const float dist = vec3_magnitude(to_ball);
  if (dist < 1.0F) return cfg_.face_ball_weight;

  const Vec3 to_ball_norm{
      to_ball.x / dist,
      to_ball.y / dist,
      to_ball.z / dist,
  };

  const float facing = vec3_dot(car.forward, to_ball_norm);
  return std::max(0.0F, facing) * cfg_.face_ball_weight;
}

float DenseRewardCalculator::air_reward(
    const CarState& car, const EnvState& env) const {
  if (cfg_.air_reward_weight <= 0.0F) return 0.0F;
  if (env.ball.position.z < cfg_.air_reward_ball_z_min) return 0.0F;
  if (car.on_ground) return 0.0F;

  return cfg_.air_reward_weight;
}

float DenseRewardCalculator::velocity_ball_to_goal(
    const CarState& car, const EnvState& env) const {
  if (cfg_.velocity_ball_to_goal_weight <= 0.0F) return 0.0F;

  const float team_sign = (car.team == Team::Blue) ? 1.0F : -1.0F;
  const float vel_toward = env.ball.velocity.y * team_sign;
  const float frac = clamp(vel_toward / std::max(cfg_.max_ball_speed, 1.0F), 0.0F, 1.0F);

  return frac * cfg_.velocity_ball_to_goal_weight;
}

float DenseRewardCalculator::air_touch(
    const CarState& car, const EnvState& env, float vel_delta_frac) const {
  if (cfg_.air_touch_weight <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;
  if (car.on_ground) return 0.0F;

  const float height_frac = clamp(env.ball.position.z / kCeilingZ, 0.0F, 1.0F);
  const float air_time_frac = clamp(
      car.air_time_since_jump / cfg_.air_touch_max_air_time, 0.0F, 1.0F);
  const float quality = std::min(height_frac, air_time_frac);

  return quality * vel_delta_frac * cfg_.air_touch_weight;
}

float DenseRewardCalculator::save_boost(const CarState& car) const {
  if (cfg_.save_boost_weight <= 0.0F) return 0.0F;

  return std::sqrt(std::max(0.0F, car.boost)) * cfg_.save_boost_weight;
}

float DenseRewardCalculator::boost_pickup(
    const CarState& car, AgentDenseState& s) const {
  if (cfg_.boost_pickup_big_weight <= 0.0F
      && cfg_.boost_pickup_small_weight <= 0.0F) {
    return 0.0F;
  }

  const float amount = car.boost - s.prev_boost;
  if (amount <= 0.001F) return 0.0F;

  if (amount > cfg_.boost_pickup_big_threshold) {
    return amount * cfg_.boost_pickup_big_weight;
  }
  return amount * cfg_.boost_pickup_small_weight;
}

}  // namespace pulsar
