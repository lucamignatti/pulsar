#include "pulsar/env/obs_builder.hpp"

#include <algorithm>
#include <cmath>

namespace pulsar {
namespace {

constexpr float kPosScale = 1.0F / 2300.0F;
constexpr float kLinVelScale = 1.0F / 2300.0F;
constexpr float kAngVelScale = 1.0F / 3.14159265F;
constexpr float kPadTimerScale = 0.1F;
constexpr float kBoostScale = 0.01F;

float norm(float value, float scale) {
  return value / scale;
}

void append_vec3_scaled(std::vector<float>* out, const Vec3& value, float coef) {
  out->push_back(value.x * coef);
  out->push_back(value.y * coef);
  out->push_back(value.z * coef);
}

Vec3 maybe_invert(const Vec3& value, bool inverted) {
  return inverted ? invert_for_orange(value) : value;
}

void append_car_obs(std::vector<float>* out, const CarState& car, bool inverted) {
  const Vec3 pos = maybe_invert(car.position, inverted);
  const Vec3 forward = maybe_invert(car.forward, inverted);
  const Vec3 up = maybe_invert(car.up, inverted);
  const Vec3 lin_vel = maybe_invert(car.velocity, inverted);
  const Vec3 ang_vel = maybe_invert(car.angular_velocity, inverted);

  append_vec3_scaled(out, pos, kPosScale);
  append_vec3_scaled(out, forward, 1.0F);
  append_vec3_scaled(out, up, 1.0F);
  append_vec3_scaled(out, lin_vel, kLinVelScale);
  append_vec3_scaled(out, ang_vel, kAngVelScale);
  out->push_back(car.boost * kBoostScale);
  out->push_back(car.demo_respawn_timer);
  out->push_back(car.on_ground ? 1.0F : 0.0F);
  out->push_back(car.is_boosting ? 1.0F : 0.0F);
  out->push_back(car.is_supersonic ? 1.0F : 0.0F);
}

}  // namespace

PulsarObsBuilder::PulsarObsBuilder(EnvConfig config) : config_(std::move(config)) {}

std::vector<float> PulsarObsBuilder::build_obs(const EnvState& state, AgentId agent_id) const {
  const CarState& self = state.cars.at(agent_id);
  const bool inverted = self.team == Team::Orange;
  std::vector<float> obs;
  obs.reserve(obs_dim());

  append_vec3_scaled(&obs, maybe_invert(state.ball.position, inverted), kPosScale);
  append_vec3_scaled(&obs, maybe_invert(state.ball.velocity, inverted), kLinVelScale);
  append_vec3_scaled(&obs, maybe_invert(state.ball.angular_velocity, inverted), kAngVelScale);
  for (float pad_timer : state.boost_pad_timers) {
    obs.push_back(pad_timer * kPadTimerScale);
  }
  obs.push_back(self.is_holding_jump ? 1.0F : 0.0F);
  obs.push_back(self.handbrake);
  obs.push_back(self.has_jumped ? 1.0F : 0.0F);
  obs.push_back(self.is_jumping ? 1.0F : 0.0F);
  obs.push_back(self.has_flipped ? 1.0F : 0.0F);
  obs.push_back(self.is_flipping ? 1.0F : 0.0F);
  obs.push_back(self.has_double_jumped ? 1.0F : 0.0F);
  obs.push_back(self.has_flip ? 1.0F : 0.0F);
  obs.push_back(self.air_time_since_jump);

  append_car_obs(&obs, self, inverted);

  std::vector<const CarState*> allies;
  std::vector<const CarState*> enemies;
  for (const auto& other : state.cars) {
    if (other.id == self.id) {
      continue;
    }
    if (other.team == self.team) {
      allies.push_back(&other);
    } else {
      enemies.push_back(&other);
    }
  }

  while (allies.size() < static_cast<std::size_t>(config_.team_size - 1)) {
    allies.push_back(nullptr);
  }
  while (enemies.size() < static_cast<std::size_t>(config_.team_size)) {
    enemies.push_back(nullptr);
  }

  for (const CarState* ally : allies) {
    if (ally == nullptr) {
      obs.insert(obs.end(), 20, 0.0F);
    } else {
      append_car_obs(&obs, *ally, inverted);
    }
  }
  for (const CarState* enemy : enemies) {
    if (enemy == nullptr) {
      obs.insert(obs.end(), 20, 0.0F);
    } else {
      append_car_obs(&obs, *enemy, inverted);
    }
  }

  return obs;
}

std::size_t PulsarObsBuilder::obs_dim() const {
  return 52 + 20 * static_cast<std::size_t>(config_.team_size) * 2;
}

}  // namespace pulsar
