#pragma once

#include <vector>
#include "pulsar/config/config.hpp"
#include "pulsar/core/types.hpp"

namespace pulsar {

struct AgentMechanicState {
  bool prev_on_ground = true;
  bool prev_has_flip = true;
  bool prev_has_flipped = false;
  bool prev_is_flipping = false;
  bool prev_has_flip_reset = false;
  bool prev_ball_touched = false;
  Vec3 prev_ball_velocity{};

  int consecutive_air_touches = 0;
  int last_wavedash_tick = -1000;
  int last_ball_touch_tick = -1000;
  int flip_start_tick = -1;

  bool was_on_ceiling = false;
  bool ball_hit_backboard = false;
  int last_toucher_id = -1;
};

struct EnvMechanicState {
  int first_touch_team = -1;
  bool kickoff_reward_given = false;
};

class MechanicDetector {
 public:
  explicit MechanicDetector(const MechanicRewardConfig& cfg);

  float update(
      int global_tick,
      const CarState& car,
      const EnvState& env,
      int env_team_size,
      AgentMechanicState& agent_state,
      EnvMechanicState& env_state) const;

 private:
  MechanicRewardConfig cfg_;

  static constexpr float kGroundZ = 17.0F;
  static constexpr float kCeilingZ = 2044.0F;
  static constexpr float kArenaExtentX = 4096.0F;
  static constexpr float kArenaExtentY = 5120.0F;
  static constexpr float kWavedashZThreshold = 70.0F;
  static constexpr float kWallDashDist = 100.0F;
  static constexpr float kSupersonicSpeed = 2190.0F;
  static constexpr float kPinchVelocityDelta = 800.0F;
  static constexpr int kChainDashWindowTicks = 30;
  static constexpr int kDoubleTapWindowTicks = 30;
  static constexpr float kCeilingThreshold = 1800.0F;

  static float vec3_dot(const Vec3& a, const Vec3& b);
  static float vec3_magnitude(const Vec3& v);

  float detect_speed_flip(const CarState& car, AgentMechanicState& s) const;
  float detect_wavedash(int tick, const CarState& car, AgentMechanicState& s) const;
  float detect_chain_dash(int tick, int prev_dash_tick, AgentMechanicState& s) const;
  float detect_half_flip(const CarState& car, AgentMechanicState& s) const;
  float detect_wall_dash(int tick, const CarState& car, AgentMechanicState& s) const;
  float detect_air_dribble(const CarState& car, AgentMechanicState& s) const;
  float detect_flip_reset(const CarState& car, AgentMechanicState& s) const;
  float detect_ceiling_shot(const CarState& car, AgentMechanicState& s) const;
  float detect_double_tap(int tick, const CarState& car, const EnvState& env, AgentMechanicState& s) const;
  float detect_preflip(const CarState& car, AgentMechanicState& s) const;
  float detect_redirect(const CarState& car, const EnvState& env, AgentMechanicState& s, Team team) const;
  float detect_pogo(const CarState& car, AgentMechanicState& s) const;
  float detect_pinch(const EnvState& env, AgentMechanicState& s) const;
  float detect_team_pinch(const CarState& car, const EnvState& env, AgentMechanicState& s, int env_team_size) const;
  float detect_kickoff_first_touch(const CarState& car, const EnvState& env,
                                   EnvMechanicState& env_state) const;
};

}  // namespace pulsar
