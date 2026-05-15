#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/core/types.hpp"

namespace pulsar {

struct AgentRewardState {
  float prev_boost = 0.0F;
  Vec3 prev_ball_velocity{};
  float episode_dense_reward = 0.0F;
  float episode_boost_pickup_reward = 0.0F;
  float episode_air_touch_reward = 0.0F;

  bool prev_on_ground = true;
  float prev_car_velocity_z = 0.0F;
  bool prev_has_flip = true;
  bool prev_has_flipped = false;
  bool prev_is_flipping = false;
  bool prev_has_flip_reset = false;
  bool prev_ball_touched = false;

  int consecutive_air_touches = 0;
  int consecutive_touches = 0;
  int last_wavedash_tick = -1000;
  int last_ball_touch_tick = -1000;
  int last_own_touch_tick = -1000;
  int prev_env_last_touch_agent = -1;
  int flip_start_tick = -1;

  bool was_on_ceiling = false;
  bool ball_hit_backboard = false;
  int backboard_hit_count = 0;
  int last_toucher_id = -1;

  float episode_mechanic_reward = 0.0F;
};

struct EnvRewardState {
  int first_touch_team = -1;
  bool kickoff_reward_given = false;
};

struct RewardBreakdown {
  float total = 0.0F;
  float terminal = 0.0F;
  float gameplay = 0.0F;
  float mechanic = 0.0F;
  std::unordered_map<std::string, float> terms;
};

class RewardEngine {
 public:
  explicit RewardEngine(const ExperimentConfig& cfg);
  void update_config(const ExperimentConfig& cfg);
  void set_unlocked_mechanics(const std::vector<std::string>& mechanics);

  RewardBreakdown compute(
      int global_tick,
      const CarState& car,
      const EnvState& env,
      int env_team_size,
      AgentRewardState& agent_state,
      EnvRewardState& env_state,
      bool done,
      int outcome_label) const;

 private:
  OutcomeConfig outcome_;
  MechanicRewardConfig mechanic_cfg_;
  DenseRewardConfig dense_cfg_;
  std::unordered_set<std::string> unlocked_mechanics_;

  // --- game constants ---
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
  // velocity_ball_to_goal: how long after a touch the agent is credited, in
  // arena ticks at 120 Hz (120 ticks ≈ 1 second of real time).
  static constexpr int kVbtgTouchWindowTicks = 120;

  bool is_mechanic_unlocked(const std::string& name) const;

  // --- terminal reward ---
  static float terminal_reward(int outcome_label, const OutcomeConfig& oc);

  // --- dense/gameplay rewards ---
  float ball_touch_vel(const CarState& car, const EnvState& env, AgentRewardState& s) const;
  float touch_direction(const CarState& car, const EnvState& env, AgentRewardState& s) const;
  float speed_toward_ball(const CarState& car, const EnvState& env) const;
  float face_ball(const CarState& car, const EnvState& env) const;
  float air_reward(const CarState& car, const EnvState& env) const;
  float velocity_ball_to_goal(const CarState& car, const EnvState& env) const;
  float air_touch(const CarState& car, const EnvState& env, float vel_delta_frac) const;
  float save_boost(const CarState& car) const;
  float boost_efficiency(const CarState& car, const EnvState& env) const;
  float boost_used(const CarState& car, AgentRewardState& s) const;
  float defensive_positioning(const CarState& car, const EnvState& env) const;
  float shot_accuracy(const CarState& car, const EnvState& env) const;
  float boost_pickup(const CarState& car, AgentRewardState& s) const;
  float possession_chain(const CarState& car, const EnvState& env, int global_tick, AgentRewardState& s) const;

  // --- mechanic rewards ---
  float detect_speed_flip(const CarState& car, AgentRewardState& s) const;
  float detect_wavedash(int tick, const CarState& car, AgentRewardState& s) const;
  float detect_chain_dash(int tick, int prev_dash_tick, AgentRewardState& s) const;
  float detect_half_flip(const CarState& car, AgentRewardState& s) const;
  float detect_wall_dash(int tick, const CarState& car, AgentRewardState& s) const;
  float detect_air_dribble(const CarState& car, AgentRewardState& s) const;
  float detect_flip_reset(const CarState& car, AgentRewardState& s) const;
  float detect_ceiling_shot(const CarState& car, AgentRewardState& s) const;
  float detect_double_tap(int tick, const CarState& car, const EnvState& env, AgentRewardState& s) const;
  float detect_preflip(const CarState& car, AgentRewardState& s) const;
  float detect_redirect(const CarState& car, const EnvState& env, AgentRewardState& s, Team team) const;
  float detect_pogo(const CarState& car, AgentRewardState& s) const;
  float detect_pinch(const CarState& car, const EnvState& env, AgentRewardState& s) const;
  float detect_team_pinch(const CarState& car, const EnvState& env, AgentRewardState& s, int env_team_size) const;
  float detect_kickoff_first_touch(const CarState& car, const EnvState& env, EnvRewardState& env_state) const;
};

}  // namespace pulsar
