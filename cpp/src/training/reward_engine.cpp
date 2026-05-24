#include "pulsar/training/reward_engine.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>

namespace pulsar {

namespace {

float vec3_dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float vec3_magnitude(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float clamp(float v, float lo, float hi) {
  return std::min(std::max(v, lo), hi);
}

void set_event(std::uint8_t* events, std::string_view name) {
  const int index = sparse_event_channel_index(name);
  if (index >= 0) {
    events[index] = 1;
  }
}

}  // namespace

RewardEngine::RewardEngine(const ExperimentConfig& cfg)
    : outcome_(cfg.outcome) {}

void RewardEngine::update_config(const ExperimentConfig& cfg) {
  outcome_ = cfg.outcome;
}

void RewardEngine::set_unlocked_mechanics(const std::vector<std::string>& mechanics) {
  // No-op
}

float RewardEngine::outcome_score() const { return outcome_.score; }
float RewardEngine::outcome_concede() const { return outcome_.concede; }
float RewardEngine::outcome_neutral() const { return outcome_.neutral; }
float RewardEngine::outcome_neutral_no_touch() const { return outcome_.neutral_no_touch; }
float RewardEngine::outcome_team_spirit() const { return outcome_.team_spirit; }
float RewardEngine::outcome_step_penalty() const { return outcome_.step_penalty; }

RewardBreakdown RewardEngine::compute(
    int global_tick,
    const CarState& car,
    const EnvState& env,
    int env_team_size,
    AgentRewardState& agent_state,
    EnvRewardState& env_state,
    bool done,
    int outcome_label) const {

  RewardBreakdown bd{};

  // --- terminal reward ---
  if (done) {
    if (outcome_label == 0) {
      bd.terminal = outcome_.score;
      bd.terms["terminal.score"] = outcome_.score;
    } else if (outcome_label == 1) {
      bd.terminal = outcome_.concede;
      bd.terms["terminal.concede"] = outcome_.concede;
    } else if (outcome_label == 2) {
      bd.terminal = outcome_.neutral;
      bd.terms["terminal.neutral"] = outcome_.neutral;
    } else if (outcome_label == 3) {
      bd.terminal = outcome_.neutral_no_touch;
      bd.terms["terminal.no_touch"] = outcome_.neutral_no_touch;
    }
  }

  bd.total = bd.terminal + bd.gameplay + bd.mechanic;

  // Apply constant step penalty (non-terminal steps)
  if (!done && outcome_.step_penalty != 0.0F) {
    bd.total += outcome_.step_penalty;
    bd.terms["step_penalty"] = outcome_.step_penalty;
  }

  return bd;
}

void RewardEngine::detect_sparse_events(
    int global_tick,
    const CarState& car,
    const EnvState& env,
    std::size_t local_agent_index,
    std::span<const AgentRewardState> previous_env_agent_states,
    AgentRewardState& agent_state,
    EnvRewardState& env_state,
    bool done,
    int outcome_label,
    std::uint8_t* events_out) const {
  (void)env_state;
  std::memset(events_out, 0, kSparseEventChannelCount);

  const AgentRewardState prev = (local_agent_index < previous_env_agent_states.size())
      ? previous_env_agent_states[local_agent_index]
      : agent_state;
  const bool touch_edge = car.ball_touched && !prev.prev_ball_touched;
  const Vec3 ball_delta{
      env.ball.velocity.x - prev.prev_ball_velocity.x,
      env.ball.velocity.y - prev.prev_ball_velocity.y,
      env.ball.velocity.z - prev.prev_ball_velocity.z,
  };
  const float ball_delta_mag = vec3_magnitude(ball_delta);
  const float post_mag = vec3_magnitude(env.ball.velocity);
  const float goal_dir = (car.team == Team::Blue) ? 1.0F : -1.0F;
  const float own_goal_dir = -goal_dir;

  if (done && outcome_label == 0) {
    set_event(events_out, "goal");
  }
  if (touch_edge) {
    set_event(events_out, "ball_touch");
    agent_state.last_own_touch_tick = global_tick;
    agent_state.consecutive_touches = prev.consecutive_touches + 1;
    if (!car.on_ground && env.ball.position.z > 120.0F) {
      set_event(events_out, "air_touch");
    }
  } else if (prev.prev_env_last_touch_agent >= 0 && env.last_touch_agent >= 0 &&
             env.last_touch_agent != car.id && prev.prev_env_last_touch_agent != car.id) {
    agent_state.consecutive_touches = 0;
  } else {
    agent_state.consecutive_touches = prev.consecutive_touches;
  }

  const float boost_delta = car.boost - prev.prev_boost;
  if (boost_delta > 1.0F) {
    set_event(events_out, "boost_pickup");
    if (boost_delta >= 50.0F) {
      set_event(events_out, "big_boost_pickup");
    }
  }

  if (!prev.prev_has_flip_reset && car.has_flip_reset) {
    set_event(events_out, "flip_reset");
  }

  if (prev.prev_on_ground && !prev.prev_is_flipping && car.is_flipping && car.is_boosting &&
      (car.position.z - kGroundZ) < 150.0F) {
    set_event(events_out, "speed_flip");
  }

  const bool landing_dash = car.on_ground && !prev.prev_on_ground && prev.prev_is_flipping &&
      (prev.prev_car_position_z - kGroundZ) < kWavedashZThreshold;
  if (landing_dash) {
    const bool is_half_flip = prev.flip_start_action_pitch < kBackflipPitchThreshold;
    const bool near_wall =
        std::abs(car.position.x) > (kArenaExtentX - kWallDashDist) ||
        std::abs(car.position.y) > (kArenaExtentY - kWallDashDist);
    if (is_half_flip) {
      set_event(events_out, "half_flip");
    } else {
      set_event(events_out, near_wall ? "wall_dash" : "wavedash");
    }
    if ((global_tick - prev.last_wavedash_tick) <= kChainDashWindowTicks) {
      set_event(events_out, "chain_dash");
    }
    agent_state.last_wavedash_tick = global_tick;
  } else {
    agent_state.last_wavedash_tick = prev.last_wavedash_tick;
  }

  if (touch_edge && car.is_flipping && ball_delta_mag > 250.0F) {
    set_event(events_out, "flick");
  }

  if (car.on_ground) {
    agent_state.consecutive_air_touches = 0;
  } else if (touch_edge) {
    agent_state.consecutive_air_touches = prev.consecutive_air_touches + 1;
    if (agent_state.consecutive_air_touches >= 2) {
      set_event(events_out, "air_dribble");
    }
  } else {
    agent_state.consecutive_air_touches = prev.consecutive_air_touches;
  }

  agent_state.was_on_ceiling = prev.was_on_ceiling || (car.on_ground && car.position.z > kOnCeilingZ);
  if (agent_state.was_on_ceiling && car.position.z < kCeilingThreshold - 100.0F &&
      touch_edge && !car.on_ground) {
    set_event(events_out, "ceiling_shot");
    agent_state.was_on_ceiling = false;
  }
  if (car.on_ground && car.position.z < kCeilingThreshold) {
    agent_state.was_on_ceiling = false;
  }

  agent_state.ball_hit_backboard = prev.ball_hit_backboard || std::abs(env.ball.position.y) > 5000.0F;
  agent_state.last_ball_touch_tick = prev.last_ball_touch_tick;
  agent_state.last_toucher_id = prev.last_toucher_id;
  if (touch_edge && agent_state.ball_hit_backboard && prev.last_toucher_id == car.id &&
      !car.on_ground &&
      (global_tick - prev.last_ball_touch_tick) > 0 &&
      (global_tick - prev.last_ball_touch_tick) <= kDoubleTapWindowTicks) {
    set_event(events_out, "double_tap");
    agent_state.ball_hit_backboard = false;
  }
  if (touch_edge) {
    agent_state.last_ball_touch_tick = global_tick;
    agent_state.last_toucher_id = car.id;
  }

  if (!prev.prev_is_flipping && car.is_flipping) {
    agent_state.flip_start_tick = global_tick;
    agent_state.flip_start_action_pitch = car.last_action.pitch;
  } else {
    agent_state.flip_start_tick = prev.flip_start_tick;
    agent_state.flip_start_action_pitch = prev.flip_start_action_pitch;
  }
  if (car.is_flipping && touch_edge && prev.flip_start_tick >= 0) {
    set_event(events_out, "preflip");
    agent_state.flip_start_tick = -1;
  }
  if (car.is_flipping && !car.on_ground && prev.flip_start_tick >= 0 &&
      prev.flip_start_action_pitch < -kFlipCancelPitchThreshold &&
      car.last_action.pitch > kFlipCancelPitchThreshold) {
    set_event(events_out, "flip_cancel");
    agent_state.flip_start_tick = -1;
  }
  if (!car.is_flipping) {
    agent_state.flip_start_tick = -1;
  }

  if (touch_edge) {
    const float prev_mag = vec3_magnitude(prev.prev_ball_velocity);
    if (prev_mag > 100.0F && post_mag > 100.0F) {
      const float dot = clamp(vec3_dot(prev.prev_ball_velocity, env.ball.velocity) / (prev_mag * post_mag), -1.0F, 1.0F);
      const float angle_deg = std::acos(dot) * (180.0F / 3.14159265358979323846F);
      if (angle_deg >= 30.0F && (env.ball.velocity.y / post_mag) * goal_dir > 0.3F) {
        set_event(events_out, "redirect");
      }
    }
  }

  if (!prev.prev_on_ground && car.on_ground && !car.is_jumping && prev.prev_car_velocity_z < -200.0F) {
    set_event(events_out, "pogo");
  }

  const float dx = car.position.x - env.ball.position.x;
  const float dy = car.position.y - env.ball.position.y;
  const float dz = car.position.z - env.ball.position.z;
  const float dist_to_ball = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (dist_to_ball <= 250.0F) {
    const float prev_ball_speed = vec3_magnitude(prev.prev_ball_velocity);
    const float post_ball_speed = vec3_magnitude(env.ball.velocity);
    if (post_ball_speed - prev_ball_speed > kPinchVelocityDelta) {
      set_event(events_out, "pinch");
    }
  }

  // --- new mechanics ---

  // save: ball threatening own goal, agent touches it
  if (touch_edge &&
      prev.prev_ball_velocity.y * own_goal_dir > kSaveVelocityThreshold &&
      env.ball.position.y * own_goal_dir > 2500.0F) {
    set_event(events_out, "save");
  }

  // clear: touch from own third sending ball away from own goal
  if (touch_edge &&
      car.position.y * own_goal_dir > kClearOwnHalfThreshold &&
      env.ball.velocity.y * own_goal_dir < -200.0F &&
      post_mag > 500.0F) {
    set_event(events_out, "clear");
  }

  // boost_steal: big pad pickup on opponent's half
  if (boost_delta >= 50.0F && car.position.y * goal_dir > 500.0F) {
    set_event(events_out, "boost_steal");
  }

  // fast_aerial: double jump (not flip) in air while boosting and moving up
  if (!prev.prev_has_double_jumped && car.has_double_jumped &&
      !car.is_flipping && !car.on_ground &&
      car.is_boosting && car.velocity.z > kFastAerialVzThreshold) {
    set_event(events_out, "fast_aerial");
  }

  // landing_recovery: clean landing — fast horizontal speed, low spin
  if (!prev.prev_on_ground && car.on_ground && !car.is_flipping && !car.is_jumping) {
    const float hspeed = std::sqrt(car.velocity.x * car.velocity.x + car.velocity.y * car.velocity.y);
    if (hspeed > kLandingRecoveryMinSpeed && vec3_magnitude(car.angular_velocity) < kLandingRecoveryMaxAngVel) {
      set_event(events_out, "landing_recovery");
    }
  }

  // powerslide_turn: rising edge of handbrake on ground with significant yaw rotation
  const bool handbrake_active = car.handbrake > 0.5F;
  if (!prev.prev_handbrake_active && handbrake_active &&
      car.on_ground && std::abs(car.angular_velocity.z) > kPowerslideMinAngVelZ) {
    set_event(events_out, "powerslide_turn");
  }
  agent_state.prev_handbrake_active = handbrake_active;

  // off_wall_touch / wall_shot / off_wall_clear
  const bool near_side_wall = std::abs(car.position.x) > (kArenaExtentX - kWallTouchDistX);
  const bool near_end_wall  = std::abs(car.position.y) > (kArenaExtentY - kWallTouchDistY);
  const bool near_any_wall  = !car.on_ground && (near_side_wall || near_end_wall);
  if (touch_edge && near_any_wall) {
    set_event(events_out, "off_wall_touch");
    if (post_mag > 0.0F && env.ball.velocity.y * goal_dir / post_mag > 0.35F) {
      set_event(events_out, "wall_shot");
    }
    if (prev.prev_ball_velocity.y * own_goal_dir > 200.0F &&
        env.ball.velocity.y * own_goal_dir < 0.0F) {
      set_event(events_out, "off_wall_clear");
    }
  }

  // backboard_save: touch while ball was near own backboard and threatening own goal
  agent_state.ball_near_own_backboard =
      prev.ball_near_own_backboard ||
      (env.ball.position.y * own_goal_dir > kOwnBackboardY);
  if (touch_edge && prev.ball_near_own_backboard &&
      prev.prev_ball_velocity.y * own_goal_dir > kSaveVelocityThreshold) {
    set_event(events_out, "backboard_save");
    agent_state.ball_near_own_backboard = false;
  }
  if (env.ball.position.y * own_goal_dir <= kOwnBackboardY - 200.0F) {
    agent_state.ball_near_own_backboard = false;
  }

  // shot: touch sending ball toward opponent goal at significant speed
  if (touch_edge && post_mag > kShotMinSpeed &&
      post_mag > 0.0F && env.ball.velocity.y * goal_dir / post_mag > kShotGoalDirThreshold) {
    set_event(events_out, "shot");
  }

  bool kickoff_opponent_touch = false;
  for (std::size_t i = 0; i < env.cars.size(); ++i) {
    if (i == local_agent_index) continue;
    const CarState& other = env.cars[i];
    const AgentRewardState& other_prev =
        (i < previous_env_agent_states.size()) ? previous_env_agent_states[i] : prev;
    if (other.team != car.team && touch_edge && other.ball_touched && !other_prev.prev_ball_touched) {
      kickoff_opponent_touch = true;
    }

    const Vec3 to_other{
        other.position.x - car.position.x,
        other.position.y - car.position.y,
        other.position.z - car.position.z,
    };
    const float distance = vec3_magnitude(to_other);
    if (distance < 240.0F) {
      const Vec3 relative_velocity{
          car.velocity.x - other.velocity.x,
          car.velocity.y - other.velocity.y,
          car.velocity.z - other.velocity.z,
      };
      const float relative_speed = vec3_magnitude(relative_velocity);
      if (relative_speed > 700.0F && (global_tick - prev.last_bump_tick) > 15) {
        set_event(events_out, "bump");
        agent_state.last_bump_tick = global_tick;
      }
    }

    if (other.team != car.team &&
        other_prev.prev_demo_respawn_timer <= 0.0F &&
        other.demo_respawn_timer > 0.0F &&
        car.is_supersonic &&
        distance < 350.0F) {
      set_event(events_out, "demo");
    }
  }
  if (kickoff_opponent_touch) {
    set_event(events_out, "kickoff_50_50");
  }
  if (agent_state.last_bump_tick != global_tick) {
    agent_state.last_bump_tick = prev.last_bump_tick;
  }

  agent_state.prev_boost = car.boost;
  agent_state.prev_demo_respawn_timer = car.demo_respawn_timer;
  agent_state.prev_ball_velocity = env.ball.velocity;
  agent_state.prev_on_ground = car.on_ground;
  agent_state.prev_car_position_z = car.position.z;
  agent_state.prev_car_velocity_z = car.velocity.z;
  agent_state.prev_has_flip = car.has_flip;
  agent_state.prev_has_flipped = car.has_flipped;
  agent_state.prev_is_flipping = car.is_flipping;
  agent_state.prev_has_flip_reset = car.has_flip_reset;
  agent_state.prev_ball_touched = car.ball_touched;
  agent_state.prev_env_last_touch_agent = env.last_touch_agent;
  agent_state.prev_has_double_jumped = car.has_double_jumped;
}

}  // namespace pulsar
