#pragma once

#include "pulsar/config/config.hpp"
#include "pulsar/core/types.hpp"

namespace pulsar {

struct AgentDenseState {
  float prev_boost = 0.0F;
  Vec3 prev_ball_velocity{};
  float episode_dense_reward = 0.0F;
};

class DenseRewardCalculator {
 public:
  explicit DenseRewardCalculator(const DenseRewardConfig& cfg);

  void update_config(const DenseRewardConfig& cfg);

  float update(
      const CarState& car,
      const EnvState& env,
      AgentDenseState& s) const;

 private:
  DenseRewardConfig cfg_;

  static constexpr float kCeilingZ = 2044.0F;
  static constexpr float kArenaMaxY = 5120.0F;

  float ball_touch_vel(const CarState& car, const EnvState& env, AgentDenseState& s) const;
  float speed_toward_ball(const CarState& car, const EnvState& env) const;
  float face_ball(const CarState& car, const EnvState& env) const;
  float air_reward(const CarState& car, const EnvState& env) const;
  float velocity_ball_to_goal(const CarState& car, const EnvState& env) const;
  float air_touch(const CarState& car, const EnvState& env, float vel_delta_frac) const;
  float save_boost(const CarState& car) const;
  float boost_pickup(const CarState& car, AgentDenseState& s) const;
};

}  // namespace pulsar
