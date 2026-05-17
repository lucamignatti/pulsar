#include "pulsar/env/obs_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace pulsar {
namespace {

constexpr float kPosScale = 1.0F / 2300.0F;
constexpr float kLinVelScale = 1.0F / 2300.0F;
constexpr float kAngVelScale = 1.0F / 3.14159265F;
constexpr float kPadTimerScale = 0.1F;
constexpr float kBoostScale = 0.01F;

Vec3 maybe_invert(const Vec3& value, bool inverted) {
  return inverted ? invert_for_orange(value) : value;
}

void append_vec3_scaled(float*& out, const Vec3& value, float coef) {
  *out++ = value.x * coef;
  *out++ = value.y * coef;
  *out++ = value.z * coef;
}

void write_car_obs(float*& out, const CarState& car, bool inverted) {
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
  *out++ = car.boost * kBoostScale;
  *out++ = car.demo_respawn_timer;
  *out++ = car.on_ground ? 1.0F : 0.0F;
  *out++ = car.is_boosting ? 1.0F : 0.0F;
  *out++ = car.is_supersonic ? 1.0F : 0.0F;
}

void zero_fill(float*& out, std::size_t count) {
  std::fill_n(out, static_cast<std::ptrdiff_t>(count), 0.0F);
  out += static_cast<std::ptrdiff_t>(count);
}

void validate_car_order(const EnvState& state, std::span<const AgentId> car_order) {
  if (car_order.empty()) return;
  if (car_order.size() != state.cars.size()) {
    throw std::invalid_argument("PulsarObsBuilder received a car order with the wrong size.");
  }
  std::uint64_t seen = 0;
  for (const AgentId other_id : car_order) {
    if (other_id >= state.cars.size() || other_id >= 64U) {
      throw std::invalid_argument("PulsarObsBuilder received an invalid car order.");
    }
    const std::uint64_t bit = 1ULL << other_id;
    if ((seen & bit) != 0U) {
      throw std::invalid_argument("PulsarObsBuilder received an invalid car order.");
    }
    seen |= bit;
  }
}

void write_agent_obs(const EnvState& state, const EnvConfig& config, AgentId agent_id, std::span<const AgentId> car_order, float* dst) {
  const CarState& self = state.cars[agent_id];
  const bool inverted = self.team == Team::Orange;

  append_vec3_scaled(dst, maybe_invert(state.ball.position, inverted), kPosScale);
  append_vec3_scaled(dst, maybe_invert(state.ball.velocity, inverted), kLinVelScale);
  append_vec3_scaled(dst, maybe_invert(state.ball.angular_velocity, inverted), kAngVelScale);
  for (const float pad_timer : state.boost_pad_timers) {
    *dst++ = pad_timer * kPadTimerScale;
  }
  *dst++ = self.is_holding_jump ? 1.0F : 0.0F;
  *dst++ = self.handbrake;
  *dst++ = self.has_jumped ? 1.0F : 0.0F;
  *dst++ = self.is_jumping ? 1.0F : 0.0F;
  *dst++ = self.has_flipped ? 1.0F : 0.0F;
  *dst++ = self.is_flipping ? 1.0F : 0.0F;
  *dst++ = self.has_double_jumped ? 1.0F : 0.0F;
  *dst++ = self.has_flip ? 1.0F : 0.0F;
  *dst++ = self.air_time_since_jump;

  write_car_obs(dst, self, inverted);

  std::size_t ally_count = 0;
  std::size_t enemy_count = 0;
  const std::size_t order_size = car_order.empty() ? state.cars.size() : car_order.size();
  for (std::size_t order_idx = 0; order_idx < order_size && ally_count < static_cast<std::size_t>(config.team_size - 1); ++order_idx) {
    const AgentId other_id = car_order.empty() ? order_idx : car_order[order_idx];
    if (other_id == agent_id) {
      continue;
    }

    const CarState& other = state.cars[other_id];
    if (other.team == self.team) {
      write_car_obs(dst, other, inverted);
      ++ally_count;
    }
  }

  while (ally_count < static_cast<std::size_t>(config.team_size - 1)) {
    zero_fill(dst, 20);
    ++ally_count;
  }

  for (std::size_t order_idx = 0; order_idx < order_size && enemy_count < static_cast<std::size_t>(config.team_size); ++order_idx) {
    const AgentId other_id = car_order.empty() ? order_idx : car_order[order_idx];
    if (other_id == agent_id) {
      continue;
    }
    const CarState& other = state.cars[other_id];
    if (other.team != self.team) {
      write_car_obs(dst, other, inverted);
      ++enemy_count;
    }
  }

  while (enemy_count < static_cast<std::size_t>(config.team_size)) {
    zero_fill(dst, 20);
    ++enemy_count;
  }
}

}  // namespace

PulsarObsBuilder::PulsarObsBuilder(EnvConfig config) : config_(std::move(config)) {}

std::vector<float> PulsarObsBuilder::build_obs(const EnvState& state, AgentId agent_id) const {
  std::vector<float> obs(obs_dim());
  write_agent_obs(state, config_, agent_id, {}, obs.data());
  return obs;
}

void PulsarObsBuilder::build_obs_batch(const EnvState& state, std::span<float> out) const {
  build_obs_batch(state, out, {});
}

void PulsarObsBuilder::build_obs_batch(const EnvState& state, std::span<float> out, std::span<const AgentId> car_order) const {
  const std::size_t stride = obs_dim();
  const std::size_t expected = state.cars.size() * stride;
  if (out.size() != expected) {
    throw std::invalid_argument("PulsarObsBuilder::build_obs_batch received an output span with the wrong size.");
  }
  validate_car_order(state, car_order);

  for (std::size_t agent_id = 0; agent_id < state.cars.size(); ++agent_id) {
    float* dst = out.data() + static_cast<std::ptrdiff_t>(agent_id * stride);
    write_agent_obs(state, config_, agent_id, car_order, dst);
  }
}

std::size_t PulsarObsBuilder::obs_dim() const {
  return 52 + 20 * static_cast<std::size_t>(config_.team_size) * 2;
}

}  // namespace pulsar
