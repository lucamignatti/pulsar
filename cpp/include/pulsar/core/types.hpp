#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pulsar {

enum class Team : int {
  Blue = 0,
  Orange = 1,
};

struct Vec3 {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
};

inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vec3 operator*(const Vec3& value, float scalar) {
  return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline Vec3 invert_for_orange(const Vec3& value) {
  return {-value.x, -value.y, value.z};
}

struct ControllerState {
  float throttle = 0.0F;
  float steer = 0.0F;
  float yaw = 0.0F;
  float pitch = 0.0F;
  float roll = 0.0F;
  bool jump = false;
  bool boost = false;
  bool handbrake = false;
};

struct CarState {
  int id = -1;
  Team team = Team::Blue;
  Vec3 position{};
  Vec3 velocity{};
  Vec3 angular_velocity{};
  Vec3 forward{1.0F, 0.0F, 0.0F};
  Vec3 up{0.0F, 0.0F, 1.0F};
  float boost = 0.0F;
  bool on_ground = true;
  bool has_flip = true;
  bool ball_touched = false;
  bool is_holding_jump = false;
  bool has_jumped = false;
  bool is_jumping = false;
  bool has_flipped = false;
  bool is_flipping = false;
  bool has_double_jumped = false;
  bool is_boosting = false;
  bool is_supersonic = false;
  float air_time_since_jump = 0.0F;
  float handbrake = 0.0F;
  float demo_respawn_timer = 0.0F;
};

struct BallState {
  Vec3 position{};
  Vec3 velocity{};
  Vec3 angular_velocity{};
};

struct EnvState {
  BallState ball{};
  std::vector<CarState> cars{};
  std::vector<float> boost_pad_timers{};
  int tick = 0;
  int blue_score = 0;
  int orange_score = 0;
  bool goal_scored = false;
  bool kickoff_pause = false;
  int last_touch_agent = -1;
  int last_touch_tick = 0;
};

struct StepResult {
  EnvState state{};
  std::vector<float> rewards{};
  std::vector<std::uint8_t> terminated{};
  std::vector<std::uint8_t> truncated{};
};

}  // namespace pulsar
