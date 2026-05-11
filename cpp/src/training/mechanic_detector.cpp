#include "pulsar/training/mechanic_detector.hpp"

#include <algorithm>
#include <cmath>

namespace pulsar {

MechanicDetector::MechanicDetector(const MechanicRewardConfig& cfg) : cfg_(cfg) {}

float MechanicDetector::vec3_dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float MechanicDetector::vec3_magnitude(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float MechanicDetector::update(
    int global_tick,
    const CarState& car,
    const EnvState& env,
    int env_team_size,
    AgentMechanicState& s,
    EnvMechanicState& env_s) const {
  float reward = 0.0F;
  const int prev_wavedash_tick = s.last_wavedash_tick;

  reward += detect_speed_flip(car, s);
  reward += detect_wavedash(global_tick, car, s);
  reward += detect_chain_dash(global_tick, prev_wavedash_tick, s);
  reward += detect_half_flip(car, s);
  reward += detect_wall_dash(global_tick, car, s);
  reward += detect_air_dribble(car, s);
  reward += detect_flip_reset(car, s);
  reward += detect_ceiling_shot(car, s);
  reward += detect_double_tap(global_tick, car, env, s);
  reward += detect_preflip(car, s);
  reward += detect_redirect(car, env, s, car.team);
  reward += detect_pogo(car, s);
  reward += detect_pinch(env, s);
  reward += detect_team_pinch(car, env, s, env_team_size);
  reward += detect_kickoff_first_touch(car, env, env_s);

  const float cap = cfg_.mechanic_reward_cap_per_episode;
  if (cap > 0.0F) {
    const float remaining = cap - s.episode_mechanic_reward;
    if (remaining <= 0.0F) {
      reward = 0.0F;
    } else if (reward > remaining) {
      reward = remaining;
    }
  }
  s.episode_mechanic_reward += reward;

  s.prev_on_ground = car.on_ground;
  s.prev_has_flip = car.has_flip;
  s.prev_has_flipped = car.has_flipped;
  s.prev_is_flipping = car.is_flipping;
  s.prev_has_flip_reset = car.has_flip_reset;
  s.prev_ball_touched = car.ball_touched;
  s.prev_ball_velocity = env.ball.velocity;

  return reward;
}

float MechanicDetector::detect_speed_flip(const CarState& car, AgentMechanicState& s) const {
  if (cfg_.speed_flip <= 0.0F) return 0.0F;
  if (car.on_ground && !s.prev_is_flipping && car.is_flipping && car.is_boosting) {
    return cfg_.speed_flip;
  }
  return 0.0F;
}

float MechanicDetector::detect_wavedash(int tick, const CarState& car, AgentMechanicState& s) const {
  if (cfg_.wavedash <= 0.0F) return 0.0F;
  if (!car.on_ground && !s.prev_is_flipping && car.is_flipping) {
    const float z_above_ground = car.position.z - kGroundZ;
    if (z_above_ground < kWavedashZThreshold) {
      s.last_wavedash_tick = tick;
      return cfg_.wavedash;
    }
  }
  return 0.0F;
}

float MechanicDetector::detect_chain_dash(int tick, int prev_dash_tick, AgentMechanicState& s) const {
  (void)s;
  if (cfg_.chain_dash_bonus <= 0.0F) return 0.0F;
  if (prev_dash_tick >= 0 && (tick - prev_dash_tick) <= kChainDashWindowTicks &&
      (tick - prev_dash_tick) > 0) {
    return cfg_.chain_dash_bonus;
  }
  return 0.0F;
}

float MechanicDetector::detect_half_flip(const CarState& car, AgentMechanicState& s) const {
  if (cfg_.half_flip <= 0.0F) return 0.0F;
  if (!car.on_ground && !s.prev_is_flipping && car.is_flipping && !car.has_double_jumped) {
    const float vel_mag = vec3_magnitude(car.velocity);
    if (vel_mag < 10.0F) return 0.0F;
    const Vec3 vel_norm = {
        car.velocity.x / vel_mag,
        car.velocity.y / vel_mag,
        car.velocity.z / vel_mag,
    };
    const float fwd_dot_vel = vec3_dot(car.forward, vel_norm);
    if (fwd_dot_vel < -0.3F && car.handbrake > 0.5F) {
      return cfg_.half_flip;
    }
  }
  return 0.0F;
}

float MechanicDetector::detect_wall_dash(int tick, const CarState& car, AgentMechanicState& s) const {
  if (cfg_.wall_dash <= 0.0F) return 0.0F;
  if (!car.on_ground && !s.prev_is_flipping && car.is_flipping) {
    const float z_above_ground = car.position.z - kGroundZ;
    if (z_above_ground < kWavedashZThreshold) {
      const bool near_wall =
          std::abs(car.position.x) > (kArenaExtentX - kWallDashDist) ||
          std::abs(car.position.y) > (kArenaExtentY - kWallDashDist);
      if (near_wall) {
        s.last_wavedash_tick = tick;
        return cfg_.wall_dash;
      }
    }
  }
  return 0.0F;
}

float MechanicDetector::detect_air_dribble(const CarState& car, AgentMechanicState& s) const {
  if (cfg_.air_dribble_base <= 0.0F) return 0.0F;

  if (car.on_ground) {
    s.consecutive_air_touches = 0;
    return 0.0F;
  }

  if (car.ball_touched && !car.on_ground) {
    if (!s.prev_ball_touched) {
      s.consecutive_air_touches++;
    }
    const int touches = s.consecutive_air_touches;
    const float scale = std::max(0.0F, static_cast<float>(touches - 1));
    return cfg_.air_dribble_base + cfg_.air_dribble_scale * scale;
  }

  return 0.0F;
}

float MechanicDetector::detect_flip_reset(const CarState& car, AgentMechanicState& s) const {
  if (cfg_.flip_reset <= 0.0F) return 0.0F;
  if (!s.prev_has_flip_reset && car.has_flip_reset) {
    return cfg_.flip_reset;
  }
  return 0.0F;
}

float MechanicDetector::detect_ceiling_shot(const CarState& car, AgentMechanicState& s) const {
  if (cfg_.ceiling_shot <= 0.0F) return 0.0F;

  if (!car.on_ground && car.position.z > kCeilingThreshold) {
    s.was_on_ceiling = true;
  }

  if (s.was_on_ceiling && car.position.z < kCeilingThreshold - 100.0F) {
    if (car.ball_touched && !car.on_ground) {
      s.was_on_ceiling = false;
      return cfg_.ceiling_shot;
    }
  }

  if (car.on_ground) {
    s.was_on_ceiling = false;
  }

  return 0.0F;
}

float MechanicDetector::detect_double_tap(int tick, const CarState& car, const EnvState& env,
                                          AgentMechanicState& s) const {
  if (cfg_.double_tap <= 0.0F) return 0.0F;

  if (car.ball_touched) {
    s.last_ball_touch_tick = tick;
    s.last_toucher_id = car.id;
  }

  if (std::abs(env.ball.position.y) > kArenaExtentY) {
    s.ball_hit_backboard = true;
  }

  if (car.ball_touched && s.ball_hit_backboard && s.last_toucher_id == car.id &&
      (tick - s.last_ball_touch_tick) <= kDoubleTapWindowTicks &&
      (tick - s.last_ball_touch_tick) > 0) {
    s.ball_hit_backboard = false;
    s.last_ball_touch_tick = tick;
    return cfg_.double_tap;
  }

  return 0.0F;
}

float MechanicDetector::detect_preflip(const CarState& car, AgentMechanicState& s) const {
  if (cfg_.preflip <= 0.0F) return 0.0F;

  if (!s.prev_is_flipping && car.is_flipping) {
    s.flip_start_tick = 1;
  }

  if (car.is_flipping && car.ball_touched && !s.prev_ball_touched) {
    if (s.flip_start_tick > 0) {
      s.flip_start_tick = -1;
      return cfg_.preflip;
    }
  }

  if (!car.is_flipping) {
    s.flip_start_tick = -1;
  }

  return 0.0F;
}

float MechanicDetector::detect_redirect(const CarState& car, const EnvState& env,
                                        AgentMechanicState& s, Team team) const {
  if (cfg_.redirect <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;

  const Vec3 prev_vel = s.prev_ball_velocity;
  const Vec3 post_vel = env.ball.velocity;
  const float prev_mag = vec3_magnitude(prev_vel);

  if (prev_mag < 100.0F) return 0.0F;

  const float goal_dir = (team == Team::Blue) ? 1.0F : -1.0F;
  const float prev_y = prev_vel.y / prev_mag;
  const float post_mag = std::max(vec3_magnitude(post_vel), 1.0e-6F);
  const float post_y = post_vel.y / post_mag;

  if (prev_y * goal_dir < 0.2F && post_y * goal_dir > 0.5F) {
    return cfg_.redirect;
  }

  return 0.0F;
}

float MechanicDetector::detect_pogo(const CarState& car, AgentMechanicState& s) const {
  if (cfg_.pogo <= 0.0F) return 0.0F;

  if (!s.prev_on_ground && car.on_ground) {
    if (!car.is_jumping && !car.has_jumped) {
      const float vz = car.velocity.z;
      if (vz > 200.0F) {
        return cfg_.pogo;
      }
    }
  }

  return 0.0F;
}

float MechanicDetector::detect_pinch(const EnvState& env, AgentMechanicState& s) const {
  if (cfg_.pinch <= 0.0F) return 0.0F;

  const float prev_mag = vec3_magnitude(s.prev_ball_velocity);
  const float post_mag = vec3_magnitude(env.ball.velocity);
  const float delta = post_mag - prev_mag;

  if (delta > kPinchVelocityDelta) {
    return cfg_.pinch;
  }

  return 0.0F;
}

float MechanicDetector::detect_team_pinch(const CarState& car, const EnvState& env,
                                          AgentMechanicState& s, int env_team_size) const {
  if (cfg_.team_pinch <= 0.0F) return 0.0F;

  const float prev_mag = vec3_magnitude(s.prev_ball_velocity);
  const float post_mag = vec3_magnitude(env.ball.velocity);
  const float delta = post_mag - prev_mag;
  if (delta < kPinchVelocityDelta) return 0.0F;

  int teammate_near_ball = 0;
  for (const auto& other_car : env.cars) {
    if (other_car.id == car.id) continue;
    if (other_car.team != car.team) continue;
    const float dx = other_car.position.x - env.ball.position.x;
    const float dy = other_car.position.y - env.ball.position.y;
    const float dz = other_car.position.z - env.ball.position.z;
    const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 350.0F) {
      teammate_near_ball++;
    }
  }

  if (teammate_near_ball >= 1) {
    return cfg_.team_pinch;
  }

  return 0.0F;
}

float MechanicDetector::detect_kickoff_first_touch(const CarState& car, const EnvState& env,
                                                   EnvMechanicState& env_s) const {
  if (cfg_.kickoff_first_touch <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;
  if (env_s.kickoff_reward_given) return 0.0F;

  const int team_int = static_cast<int>(car.team);

  bool opponent_touching = false;
  for (const auto& other_car : env.cars) {
    if (other_car.id == car.id) continue;
    if (other_car.ball_touched && static_cast<int>(other_car.team) != team_int) {
      opponent_touching = true;
      break;
    }
  }

  if (opponent_touching) {
    env_s.first_touch_team = 2;
  } else {
    env_s.first_touch_team = team_int;
  }
  env_s.kickoff_reward_given = true;

  if (env_s.first_touch_team == 2) {
    return cfg_.kickoff_first_touch;
  } else if (env_s.first_touch_team == team_int) {
    return cfg_.kickoff_first_touch;
  } else {
    return -cfg_.kickoff_first_touch;
  }
}

}  // namespace pulsar
