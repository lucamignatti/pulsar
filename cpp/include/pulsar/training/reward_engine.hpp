#pragma once

#include <cstdint>
#include <string>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/core/types.hpp"
#include "pulsar/training/sparse_event_channels.hpp"

namespace pulsar {

struct AgentRewardState {
  float prev_boost = 0.0F;
  float prev_demo_respawn_timer = 0.0F;
  Vec3 prev_ball_velocity{};
  bool prev_on_ground = true;
  float prev_car_position_z = 0.0F;
  float prev_car_velocity_z = 0.0F;
  bool prev_has_flip = true;
  bool prev_has_flipped = false;
  bool prev_is_flipping = false;
  bool prev_ball_touched = false;
  bool prev_has_flip_reset = false;
  int consecutive_air_touches = 0;
  int consecutive_touches = 0;
  int last_wavedash_tick = -1000;
  int last_ball_touch_tick = -1000;
  int last_own_touch_tick = -1000;
  int prev_env_last_touch_agent = -1;
  int flip_start_tick = -1;
  int last_bump_tick = -1000;
  bool was_on_ceiling = false;
  bool ball_hit_backboard = false;
  int last_toucher_id = -1;
};

struct EnvRewardState {
  // Empty, but kept for backward compatibility
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

  [[nodiscard]] bool has_any_gameplay_reward() const { return false; }
  [[nodiscard]] bool has_any_mechanic_reward() const { return false; }
  [[nodiscard]] bool has_team_spirit() const { return false; }
  [[nodiscard]] bool has_velocity_ball_to_goal() const { return false; }

  [[nodiscard]] float outcome_score() const;
  [[nodiscard]] float outcome_concede() const;
  [[nodiscard]] float outcome_neutral() const;
  [[nodiscard]] float outcome_neutral_no_touch() const;

  RewardBreakdown compute(
      int global_tick,
      const CarState& car,
      const EnvState& env,
      int env_team_size,
      AgentRewardState& agent_state,
      EnvRewardState& env_state,
      bool done,
      int outcome_label) const;

  void detect_sparse_events(
      int global_tick,
      const CarState& car,
      const EnvState& env,
      std::size_t local_agent_index,
      std::span<const AgentRewardState> previous_env_agent_states,
      AgentRewardState& agent_state,
      EnvRewardState& env_state,
      bool done,
      int outcome_label,
      std::uint8_t* events_out) const;

 private:
  OutcomeConfig outcome_;

  static constexpr float kGroundZ = 17.0F;
  static constexpr float kCeilingZ = 2044.0F;
  static constexpr float kArenaExtentX = 4096.0F;
  static constexpr float kArenaExtentY = 5120.0F;
  static constexpr float kWavedashZThreshold = 70.0F;
  static constexpr float kWallDashDist = 100.0F;
  static constexpr float kPinchVelocityDelta = 800.0F;
  static constexpr int kChainDashWindowTicks = 30;
  static constexpr int kDoubleTapWindowTicks = 30;
  static constexpr float kCeilingThreshold = 1800.0F;
};

}  // namespace pulsar
