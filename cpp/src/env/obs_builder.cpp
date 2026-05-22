#include "pulsar/env/obs_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>

namespace pulsar {
extern std::vector<int> g_boost_pad_mirror_indices;
extern std::vector<Vec3> g_boost_pad_positions;
extern std::vector<bool> g_boost_pad_is_big;

inline Vec3 cross_product(const Vec3& a, const Vec3& b) {
  return {
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

inline float dot_product(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float magnitude(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 normalize(const Vec3& v) {
  float len = magnitude(v);
  if (len < 1.0e-6F) {
    return {1.0F, 0.0F, 0.0F};
  }
  return v * (1.0F / len);
}

namespace {

constexpr float kPosScale = 1.0F / 2300.0F;
constexpr float kLinVelScale = 1.0F / 2300.0F;
constexpr float kAngVelScale = 1.0F / 3.14159265F;
constexpr float kPadTimerScale = 0.1F;
constexpr float kBoostScale = 0.01F;

Vec3 maybe_invert(const Vec3& value, bool inverted) {
  return inverted ? invert_for_orange(value) : value;
}

Vec3 maybe_mirror_vector(const Vec3& value, bool mirror) {
  if (mirror) {
    return {-value.x, value.y, value.z};
  }
  return value;
}

Vec3 maybe_mirror_angular_velocity(const Vec3& value, bool mirror) {
  if (mirror) {
    return {value.x, -value.y, -value.z};
  }
  return value;
}

void append_vec3_scaled(float*& out, const Vec3& value, float coef) {
  *out++ = value.x * coef;
  *out++ = value.y * coef;
  *out++ = value.z * coef;
}

void write_car_obs(float*& out, const CarState& car, bool inverted, bool mirror) {
  const Vec3 pos = maybe_mirror_vector(maybe_invert(car.position, inverted), mirror);
  const Vec3 forward = maybe_mirror_vector(maybe_invert(car.forward, inverted), mirror);
  const Vec3 up = maybe_mirror_vector(maybe_invert(car.up, inverted), mirror);
  const Vec3 lin_vel = maybe_mirror_vector(maybe_invert(car.velocity, inverted), mirror);
  const Vec3 ang_vel = maybe_mirror_angular_velocity(maybe_invert(car.angular_velocity, inverted), mirror);

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
  float* start_dst = dst;
  const CarState& self = state.cars[agent_id];
  const bool inverted = self.team == Team::Orange;
  const Vec3 self_pos = maybe_invert(self.position, inverted);
  const bool mirror = config.obs_x_mirror && (self_pos.x > 0.0F);

  append_vec3_scaled(dst, maybe_mirror_vector(maybe_invert(state.ball.position, inverted), mirror), kPosScale);
  append_vec3_scaled(dst, maybe_mirror_vector(maybe_invert(state.ball.velocity, inverted), mirror), kLinVelScale);
  append_vec3_scaled(dst, maybe_mirror_angular_velocity(maybe_invert(state.ball.angular_velocity, inverted), mirror), kAngVelScale);
  if (mirror && !g_boost_pad_mirror_indices.empty() && g_boost_pad_mirror_indices.size() == state.boost_pad_timers.size()) {
    for (std::size_t i = 0; i < state.boost_pad_timers.size(); ++i) {
      const int mirrored_idx = g_boost_pad_mirror_indices[i];
      *dst++ = state.boost_pad_timers[static_cast<std::size_t>(mirrored_idx)] * kPadTimerScale;
    }
  } else {
    for (const float pad_timer : state.boost_pad_timers) {
      *dst++ = pad_timer * kPadTimerScale;
    }
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

  // The acting agent is always the fixed self slot; car_order only randomizes other cars.
  write_car_obs(dst, self, inverted, mirror);

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
      write_car_obs(dst, other, inverted, mirror);
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
      write_car_obs(dst, other, inverted, mirror);
      ++enemy_count;
    }
  }

  while (enemy_count < static_cast<std::size_t>(config.team_size)) {
    zero_fill(dst, 20);
    ++enemy_count;
  }

  // 1. Local-Frame Projection
  if (config.obs_local_frame) {
    const Vec3 self_f = maybe_mirror_vector(maybe_invert(self.forward, inverted), mirror);
    const Vec3 self_u = maybe_mirror_vector(maybe_invert(self.up, inverted), mirror);
    const Vec3 self_r = cross_product(self_f, self_u);
    
    const Vec3 ball_pos_im = maybe_mirror_vector(maybe_invert(state.ball.position, inverted), mirror);
    const Vec3 ball_vel_im = maybe_mirror_vector(maybe_invert(state.ball.velocity, inverted), mirror);
    const Vec3 self_pos_im = maybe_mirror_vector(maybe_invert(self.position, inverted), mirror);
    const Vec3 self_vel_im = maybe_mirror_vector(maybe_invert(self.velocity, inverted), mirror);
    
    const Vec3 rel_ball_pos = ball_pos_im - self_pos_im;
    const Vec3 rel_ball_vel = ball_vel_im - self_vel_im;
    
    Vec3 local_ball_pos = {
      dot_product(rel_ball_pos, self_r),
      dot_product(rel_ball_pos, self_f),
      dot_product(rel_ball_pos, self_u)
    };
    Vec3 local_ball_vel = {
      dot_product(rel_ball_vel, self_r),
      dot_product(rel_ball_vel, self_f),
      dot_product(rel_ball_vel, self_u)
    };
    
    append_vec3_scaled(dst, local_ball_pos, kPosScale);
    append_vec3_scaled(dst, local_ball_vel, kLinVelScale);
    
    std::size_t ally_count_local = 0;
    for (std::size_t order_idx = 0; order_idx < order_size && ally_count_local < static_cast<std::size_t>(config.team_size - 1); ++order_idx) {
      const AgentId other_id = car_order.empty() ? order_idx : car_order[order_idx];
      if (other_id == agent_id) continue;
      
      const CarState& other = state.cars[other_id];
      if (other.team == self.team) {
        const Vec3 other_pos_im = maybe_mirror_vector(maybe_invert(other.position, inverted), mirror);
        const Vec3 other_vel_im = maybe_mirror_vector(maybe_invert(other.velocity, inverted), mirror);
        const Vec3 rel_other_pos = other_pos_im - self_pos_im;
        const Vec3 rel_other_vel = other_vel_im - self_vel_im;
        
        Vec3 local_other_pos = {
          dot_product(rel_other_pos, self_r),
          dot_product(rel_other_pos, self_f),
          dot_product(rel_other_pos, self_u)
        };
        Vec3 local_other_vel = {
          dot_product(rel_other_vel, self_r),
          dot_product(rel_other_vel, self_f),
          dot_product(rel_other_vel, self_u)
        };
        
        append_vec3_scaled(dst, local_other_pos, kPosScale);
        append_vec3_scaled(dst, local_other_vel, kLinVelScale);
        ++ally_count_local;
      }
    }
    while (ally_count_local < static_cast<std::size_t>(config.team_size - 1)) {
      zero_fill(dst, 6);
      ++ally_count_local;
    }
    
    std::size_t enemy_count_local = 0;
    for (std::size_t order_idx = 0; order_idx < order_size && enemy_count_local < static_cast<std::size_t>(config.team_size); ++order_idx) {
      const AgentId other_id = car_order.empty() ? order_idx : car_order[order_idx];
      if (other_id == agent_id) continue;
      
      const CarState& other = state.cars[other_id];
      if (other.team != self.team) {
        const Vec3 other_pos_im = maybe_mirror_vector(maybe_invert(other.position, inverted), mirror);
        const Vec3 other_vel_im = maybe_mirror_vector(maybe_invert(other.velocity, inverted), mirror);
        const Vec3 rel_other_pos = other_pos_im - self_pos_im;
        const Vec3 rel_other_vel = other_vel_im - self_vel_im;
        
        Vec3 local_other_pos = {
          dot_product(rel_other_pos, self_r),
          dot_product(rel_other_pos, self_f),
          dot_product(rel_other_pos, self_u)
        };
        Vec3 local_other_vel = {
          dot_product(rel_other_vel, self_r),
          dot_product(rel_other_vel, self_f),
          dot_product(rel_other_vel, self_u)
        };
        
        append_vec3_scaled(dst, local_other_pos, kPosScale);
        append_vec3_scaled(dst, local_other_vel, kLinVelScale);
        ++enemy_count_local;
      }
    }
    while (enemy_count_local < static_cast<std::size_t>(config.team_size)) {
      zero_fill(dst, 6);
      ++enemy_count_local;
    }
  }

  // 2. Relative Goals
  if (config.obs_relative_goals) {
    const Vec3 self_pos_im = maybe_mirror_vector(maybe_invert(self.position, inverted), mirror);
    
    const Vec3 opp_goal_center = {0.0F, 5120.0F, 0.0F};
    const Vec3 opp_goal_left = {-892.8F, 5120.0F, 0.0F};
    const Vec3 opp_goal_right = {892.8F, 5120.0F, 0.0F};
    
    const Vec3 own_goal_center = {0.0F, -5120.0F, 0.0F};
    const Vec3 own_goal_left = {-892.8F, -5120.0F, 0.0F};
    const Vec3 own_goal_right = {892.8F, -5120.0F, 0.0F};
    
    append_vec3_scaled(dst, opp_goal_center - self_pos_im, kPosScale);
    append_vec3_scaled(dst, opp_goal_left - self_pos_im, kPosScale);
    append_vec3_scaled(dst, opp_goal_right - self_pos_im, kPosScale);
    
    append_vec3_scaled(dst, own_goal_center - self_pos_im, kPosScale);
    append_vec3_scaled(dst, own_goal_left - self_pos_im, kPosScale);
    append_vec3_scaled(dst, own_goal_right - self_pos_im, kPosScale);
  }

  // 3. Proximity Boosts
  if (config.obs_proximity_boosts) {
    const Vec3 self_pos_im = maybe_mirror_vector(maybe_invert(self.position, inverted), mirror);
    
    struct PadDist {
      Vec3 pos_im;
      float timer = 0.0F;
      float dist = 0.0F;
    };
    std::vector<PadDist> small_pads;
    std::vector<PadDist> big_pads;
    
    for (std::size_t i = 0; i < g_boost_pad_positions.size() && i < state.boost_pad_timers.size(); ++i) {
      const Vec3 pad_pos = g_boost_pad_positions[i];
      const bool is_big = g_boost_pad_is_big[i];
      
      const Vec3 pad_pos_im = maybe_mirror_vector(maybe_invert(pad_pos, inverted), mirror);
      const float dist = magnitude(pad_pos_im - self_pos_im);
      const float timer = state.boost_pad_timers[i];
      
      if (is_big) {
        big_pads.push_back({pad_pos_im, timer, dist});
      } else {
        small_pads.push_back({pad_pos_im, timer, dist});
      }
    }
    
    std::sort(small_pads.begin(), small_pads.end(), [](const PadDist& a, const PadDist& b) {
      return a.dist < b.dist;
    });
    std::sort(big_pads.begin(), big_pads.end(), [](const PadDist& a, const PadDist& b) {
      return a.dist < b.dist;
    });
    
    for (std::size_t i = 0; i < 3; ++i) {
      if (i < small_pads.size()) {
        append_vec3_scaled(dst, small_pads[i].pos_im - self_pos_im, kPosScale);
        *dst++ = small_pads[i].timer * kPadTimerScale;
      } else {
        zero_fill(dst, 4);
      }
    }
    for (std::size_t i = 0; i < 2; ++i) {
      if (i < big_pads.size()) {
        append_vec3_scaled(dst, big_pads[i].pos_im - self_pos_im, kPosScale);
        *dst++ = big_pads[i].timer * kPadTimerScale;
      } else {
        zero_fill(dst, 4);
      }
    }
  }

  // 4. Explicit Kinematics
  if (config.obs_explicit_kinematics) {
    const Vec3 ball_pos_im = maybe_mirror_vector(maybe_invert(state.ball.position, inverted), mirror);
    const Vec3 ball_vel_im = maybe_mirror_vector(maybe_invert(state.ball.velocity, inverted), mirror);
    const Vec3 self_pos_im = maybe_mirror_vector(maybe_invert(self.position, inverted), mirror);
    const Vec3 self_vel_im = maybe_mirror_vector(maybe_invert(self.velocity, inverted), mirror);
    
    const Vec3 rel_pos = ball_pos_im - self_pos_im;
    const Vec3 rel_vel = ball_vel_im - self_vel_im;
    
    const float dist = magnitude(rel_pos);
    float closing_vel = 0.0F;
    if (dist > 1.0e-6F) {
      closing_vel = -dot_product(rel_pos, rel_vel) / dist;
    }
    
    *dst++ = dist * kPosScale;
    *dst++ = closing_vel * kLinVelScale;
  }

  // 5. Flip Decay
  if (config.obs_flip_decay) {
    float flip_timer = 0.0F;
    if (self.has_flip) {
      if (self.has_flip_reset) {
        flip_timer = 1.0F;
      } else if (self.has_jumped) {
        flip_timer = std::max(0.0F, 1.25F - self.air_time_since_jump) / 1.25F;
      } else {
        flip_timer = 1.0F;
      }
    }
    *dst++ = flip_timer;
  }

  // 6. Action History
  if (config.obs_action_history) {
    *dst++ = self.last_action.throttle;
    *dst++ = self.last_action.steer;
    *dst++ = self.last_action.yaw;
    *dst++ = self.last_action.pitch;
    *dst++ = self.last_action.roll;
    *dst++ = self.last_action.jump ? 1.0F : 0.0F;
    *dst++ = self.last_action.boost ? 1.0F : 0.0F;
    *dst++ = self.last_action.handbrake ? 1.0F : 0.0F;
  }

  // 7. Ball Prediction
  if (config.obs_ball_prediction) {
    const Vec3 ball_pos_im = maybe_mirror_vector(maybe_invert(state.ball.position, inverted), mirror);
    const Vec3 ball_vel_im = maybe_mirror_vector(maybe_invert(state.ball.velocity, inverted), mirror);
    const Vec3 self_pos_im = maybe_mirror_vector(maybe_invert(self.position, inverted), mirror);
    
    float dt = 1.0F / 60.0F;
    Vec3 p = ball_pos_im;
    Vec3 v = ball_vel_im;
    Vec3 p_025{}, p_05{}, p_10{};
    for (int step = 1; step <= 60; ++step) {
      v = v * (1.0F - 0.03F * dt);
      v.z += -650.0F * dt;
      p = p + v * dt;
      
      // Bounces
      if (p.z < 91.25F) {
        p.z = 91.25F;
        v.z = -v.z * 0.6F;
      } else if (p.z > 1956.75F) {
        p.z = 1956.75F;
        v.z = -v.z * 0.6F;
      }
      if (p.x < -4004.75F) {
        p.x = -4004.75F;
        v.x = -v.x * 0.6F;
      } else if (p.x > 4004.75F) {
        p.x = 4004.75F;
        v.x = -v.x * 0.6F;
      }
      if (p.y < -5028.75F) {
        p.y = -5028.75F;
        v.y = -v.y * 0.6F;
      } else if (p.y > 5028.75F) {
        p.y = 5028.75F;
        v.y = -v.y * 0.6F;
      }
      
      if (step == 15) p_025 = p;
      if (step == 30) p_05 = p;
      if (step == 60) p_10 = p;
    }
    
    append_vec3_scaled(dst, p_025 - self_pos_im, kPosScale);
    append_vec3_scaled(dst, p_05 - self_pos_im, kPosScale);
    append_vec3_scaled(dst, p_10 - self_pos_im, kPosScale);
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
  std::size_t base = 52 + 20 * static_cast<std::size_t>(config_.team_size) * 2;
  if (config_.obs_local_frame) {
    base += 12 * static_cast<std::size_t>(config_.team_size);
  }
  if (config_.obs_relative_goals) {
    base += 18;
  }
  if (config_.obs_proximity_boosts) {
    base += 20;
  }
  if (config_.obs_explicit_kinematics) {
    base += 2;
  }
  if (config_.obs_flip_decay) {
    base += 1;
  }
  if (config_.obs_action_history) {
    base += 8;
  }
  if (config_.obs_ball_prediction) {
    base += 9;
  }
  return base;
}

}  // namespace pulsar
