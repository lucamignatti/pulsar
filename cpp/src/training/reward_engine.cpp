#include "pulsar/training/reward_engine.hpp"

#include <algorithm>
#include <cmath>

namespace pulsar {

namespace {

static float vec3_dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float vec3_magnitude(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static float safe_division(float n, float d) {
  return n / std::max(d, 1.0e-6F);
}

static float clamp(float v, float lo, float hi) {
  return std::min(std::max(v, lo), hi);
}

}  // namespace

RewardEngine::RewardEngine(const ExperimentConfig& cfg)
    : outcome_(cfg.outcome),
      mechanic_cfg_(cfg.mechanic_rewards),
      dense_cfg_(cfg.dense_rewards) {}

void RewardEngine::update_config(const ExperimentConfig& cfg) {
  outcome_ = cfg.outcome;
  mechanic_cfg_ = cfg.mechanic_rewards;
  dense_cfg_ = cfg.dense_rewards;
}

void RewardEngine::set_unlocked_mechanics(const std::vector<std::string>& mechanics) {
  unlocked_mechanics_.clear();
  for (const auto& m : mechanics) {
    unlocked_mechanics_.insert(m);
  }
}

bool RewardEngine::has_any_gameplay_reward() const {
  return dense_cfg_.ball_touch_vel_weight > 0.0F ||
         dense_cfg_.flat_touch_weight > 0.0F ||
         dense_cfg_.touch_direction_weight > 0.0F ||
         dense_cfg_.speed_toward_ball_weight > 0.0F ||
         dense_cfg_.face_ball_weight > 0.0F ||
         dense_cfg_.air_reward_weight > 0.0F ||
         dense_cfg_.velocity_ball_to_goal_weight > 0.0F ||
         dense_cfg_.air_touch_weight > 0.0F ||
         dense_cfg_.save_boost_weight > 0.0F ||
         dense_cfg_.boost_efficiency_weight > 0.0F ||
         dense_cfg_.boost_used_weight > 0.0F ||
         dense_cfg_.defensive_positioning_weight > 0.0F ||
         dense_cfg_.shot_accuracy_weight > 0.0F ||
         dense_cfg_.boost_pickup_big_weight > 0.0F ||
         dense_cfg_.boost_pickup_small_weight > 0.0F ||
         dense_cfg_.possession_chain_weight > 0.0F ||
         dense_cfg_.possession_proximity_weight > 0.0F ||
         dense_cfg_.possession_speed_toward_ball_weight > 0.0F ||
         dense_cfg_.possession_face_ball_weight > 0.0F;
}

bool RewardEngine::has_any_mechanic_reward() const {
  return mechanic_cfg_.kickoff_first_touch > 0.0F ||
         mechanic_cfg_.kickoff_50_50_touch > 0.0F ||
         mechanic_cfg_.speed_flip > 0.0F ||
         mechanic_cfg_.wavedash > 0.0F ||
         mechanic_cfg_.chain_dash_bonus > 0.0F ||
         mechanic_cfg_.half_flip > 0.0F ||
         mechanic_cfg_.wall_dash > 0.0F ||
         mechanic_cfg_.air_dribble_base > 0.0F ||
         mechanic_cfg_.flip_reset > 0.0F ||
         mechanic_cfg_.ceiling_shot > 0.0F ||
         mechanic_cfg_.double_tap > 0.0F ||
         mechanic_cfg_.preflip > 0.0F ||
         mechanic_cfg_.redirect > 0.0F ||
         mechanic_cfg_.pogo > 0.0F ||
         mechanic_cfg_.pinch > 0.0F ||
         mechanic_cfg_.team_pinch > 0.0F;
}

bool RewardEngine::has_team_spirit() const {
  return dense_cfg_.team_spirit > 0.0F;
}

bool RewardEngine::has_velocity_ball_to_goal() const {
  return dense_cfg_.velocity_ball_to_goal_weight > 0.0F;
}

float RewardEngine::outcome_score() const { return outcome_.score; }
float RewardEngine::outcome_concede() const { return outcome_.concede; }
float RewardEngine::outcome_neutral() const { return outcome_.neutral; }
float RewardEngine::outcome_neutral_no_touch() const { return outcome_.neutral_no_touch; }

bool RewardEngine::is_mechanic_unlocked(const std::string& name) const {
  if (unlocked_mechanics_.empty()) return true;  // not configured -> all unlocked (backward compat)
  return unlocked_mechanics_.count(name) > 0;
}

// --- terminal reward -----------------------------------------------------------

float RewardEngine::terminal_reward(int outcome_label, const OutcomeConfig& oc) {
  if (outcome_label == 0) return oc.score;
  if (outcome_label == 1) return oc.concede;
  if (outcome_label == 2) return oc.neutral;
  if (outcome_label == 3) return oc.neutral_no_touch;
  return 0.0F;
}

// --- compute -------------------------------------------------------------------

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
    bd.terminal = terminal_reward(outcome_label, outcome_);
    if (outcome_label == 0) {
      bd.terms["terminal.score"] = outcome_.score;
    } else if (outcome_label == 1) {
      bd.terms["terminal.concede"] = outcome_.concede;
    } else if (outcome_label == 2) {
      bd.terms["terminal.neutral"] = outcome_.neutral;
    } else if (outcome_label == 3) {
      bd.terms["terminal.no_touch"] = outcome_.neutral_no_touch;
    }
  }

  // --- gameplay (dense) rewards ---
  float bt_vel = ball_touch_vel(car, env, agent_state);
  bd.terms["gameplay.ball_touch_vel"] = bt_vel;
  bd.gameplay += bt_vel;

  float ft = flat_touch(car, agent_state);
  bd.terms["gameplay.flat_touch"] = ft;
  bd.gameplay += ft;

  float td = touch_direction(car, env, agent_state);
  bd.terms["gameplay.touch_direction"] = td;
  bd.gameplay += td;

  float stb = speed_toward_ball(car, env);
  bd.terms["gameplay.speed_toward_ball"] = stb;
  bd.gameplay += stb;

  float fb = face_ball(car, env);
  bd.terms["gameplay.face_ball"] = fb;
  bd.gameplay += fb;

  float ar = air_reward(car, env);
  bd.terms["gameplay.air"] = ar;
  bd.gameplay += ar;

  float vbtg = velocity_ball_to_goal(car, env);
  bd.terms["gameplay.velocity_ball_to_goal"] = vbtg;
  bd.gameplay += vbtg;

  float vel_delta_frac = safe_division(
      bt_vel + td,
      dense_cfg_.ball_touch_vel_weight + dense_cfg_.touch_direction_weight + 1.0e-6F);
  float at = air_touch(car, env, vel_delta_frac);
  bd.terms["gameplay.air_touch"] = at;

  // per-term cap: air_touch
  if (dense_cfg_.air_touch_cap_per_episode > 0.0F) {
    float remaining = dense_cfg_.air_touch_cap_per_episode - agent_state.episode_air_touch_reward;
    if (remaining <= 0.0F) {
      at = 0.0F;
    } else if (at > remaining) {
      at = remaining;
    }
  }
  agent_state.episode_air_touch_reward += at;
  bd.gameplay += at;

  float sb = save_boost(car);
  bd.terms["gameplay.save_boost"] = sb;
  bd.gameplay += sb;

  float be = boost_efficiency(car, env);
  bd.terms["gameplay.boost_efficiency"] = be;
  bd.gameplay += be;

  float bu = boost_used(car, agent_state);
  bd.terms["gameplay.boost_used"] = bu;
  bd.gameplay += bu;

  float dp = defensive_positioning(car, env);
  bd.terms["gameplay.defensive_positioning"] = dp;
  bd.gameplay += dp;

  float sa = shot_accuracy(car, env);
  bd.terms["gameplay.shot_accuracy"] = sa;
  bd.gameplay += sa;

  float bp = boost_pickup(car, agent_state);
  bd.terms["gameplay.boost_pickup"] = bp;

  // per-term cap: boost_pickup
  if (dense_cfg_.boost_pickup_cap_per_episode > 0.0F) {
    float remaining = dense_cfg_.boost_pickup_cap_per_episode - agent_state.episode_boost_pickup_reward;
    if (remaining <= 0.0F) {
      bp = 0.0F;
    } else if (bp > remaining) {
      bp = remaining;
    }
  }
  agent_state.episode_boost_pickup_reward += bp;
  bd.gameplay += bp;

  float pc = possession_chain(car, env, global_tick, agent_state);
  bd.terms["gameplay.possession_chain"] = pc;
  bd.gameplay += pc;

  // --- mechanic rewards ---
  const int prev_wavedash_tick = agent_state.last_wavedash_tick;

  float msf = detect_speed_flip(car, agent_state);
  bd.terms["mechanic.speed_flip"] = msf;
  bd.mechanic += msf;

  float mwd = detect_wavedash(global_tick, car, agent_state);
  bd.terms["mechanic.wavedash"] = mwd;
  bd.mechanic += mwd;

  float mcd = detect_chain_dash(global_tick, prev_wavedash_tick, agent_state);
  bd.terms["mechanic.chain_dash"] = mcd;
  bd.mechanic += mcd;

  float mhf = detect_half_flip(car, agent_state);
  bd.terms["mechanic.half_flip"] = mhf;
  bd.mechanic += mhf;

  float mwad = detect_wall_dash(global_tick, car, agent_state);
  bd.terms["mechanic.wall_dash"] = mwad;
  bd.mechanic += mwad;

  float mad = detect_air_dribble(car, agent_state);
  bd.terms["mechanic.air_dribble"] = mad;
  bd.mechanic += mad;

  float mfr = detect_flip_reset(car, agent_state);
  bd.terms["mechanic.flip_reset"] = mfr;
  bd.mechanic += mfr;

  float mcs = detect_ceiling_shot(car, agent_state);
  bd.terms["mechanic.ceiling_shot"] = mcs;
  bd.mechanic += mcs;

  float mdt = detect_double_tap(global_tick, car, env, agent_state);
  bd.terms["mechanic.double_tap"] = mdt;
  bd.mechanic += mdt;

  float mpf = detect_preflip(car, agent_state);
  bd.terms["mechanic.preflip"] = mpf;
  bd.mechanic += mpf;

  float mrd = detect_redirect(car, env, agent_state, car.team);
  bd.terms["mechanic.redirect"] = mrd;
  bd.mechanic += mrd;

  float mpg = detect_pogo(car, agent_state);
  bd.terms["mechanic.pogo"] = mpg;
  bd.mechanic += mpg;

  float mpn = detect_pinch(car, env, agent_state);
  bd.terms["mechanic.pinch"] = mpn;
  bd.mechanic += mpn;

  float mtp = detect_team_pinch(car, env, agent_state, env_team_size);
  bd.terms["mechanic.team_pinch"] = mtp;
  bd.mechanic += mtp;

  float mkft = detect_kickoff_first_touch(car, env, env_state);
  bd.terms["mechanic.kickoff_first_touch"] = mkft;
  bd.mechanic += mkft;

  float mkft50 = detect_kickoff_50_50_touch(car, env, env_state);
  bd.terms["mechanic.kickoff_50_50_touch"] = mkft50;
  bd.mechanic += mkft50;

  // --- compute total ---
  bd.total = bd.terminal + bd.gameplay + bd.mechanic;

  // --- state bookkeeping ---
  agent_state.prev_on_ground = car.on_ground;
  agent_state.prev_car_position_z = car.position.z;
  agent_state.prev_car_velocity_z = car.velocity.z;
  agent_state.prev_has_flip = car.has_flip;
  agent_state.prev_has_flipped = car.has_flipped;
  agent_state.prev_is_flipping = car.is_flipping;
  agent_state.prev_has_flip_reset = car.has_flip_reset;
  agent_state.prev_ball_touched = car.ball_touched;
  agent_state.prev_ball_velocity = env.ball.velocity;
  agent_state.prev_boost = car.boost;
  agent_state.prev_env_last_touch_agent = env.last_touch_agent;

  return bd;
}

// --- gameplay / dense reward methods -------------------------------------------

float RewardEngine::ball_touch_vel(
    const CarState& car, const EnvState& env, AgentRewardState& s) const {
  if (dense_cfg_.ball_touch_vel_weight <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;

  const Vec3 delta{
      env.ball.velocity.x - s.prev_ball_velocity.x,
      env.ball.velocity.y - s.prev_ball_velocity.y,
      env.ball.velocity.z - s.prev_ball_velocity.z,
  };
  const float delta_mag = vec3_magnitude(delta);
  return clamp(delta_mag / std::max(dense_cfg_.max_ball_speed, 1.0F), 0.0F, 1.0F)
      * dense_cfg_.ball_touch_vel_weight;
}

float RewardEngine::flat_touch(
    const CarState& car, AgentRewardState& s) const {
  if (dense_cfg_.flat_touch_weight <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;
  if (s.prev_ball_touched) return 0.0F;
  return dense_cfg_.flat_touch_weight;
}

float RewardEngine::touch_direction(
    const CarState& car, const EnvState& env, AgentRewardState& s) const {
  if (dense_cfg_.touch_direction_weight <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;

  const Vec3 delta{
      env.ball.velocity.x - s.prev_ball_velocity.x,
      env.ball.velocity.y - s.prev_ball_velocity.y,
      env.ball.velocity.z - s.prev_ball_velocity.z,
  };
  const float delta_mag = vec3_magnitude(delta);
  if (delta_mag < 0.01F) return 0.0F;

  const float goal_dir = (car.team == Team::Blue) ? 1.0F : -1.0F;
  const float toward_goal = (delta.y * goal_dir) / delta_mag;

  // zero reward if hitting toward own goal
  if (toward_goal <= 0.0F) return 0.0F;

  const float delta_frac = clamp(delta_mag / std::max(dense_cfg_.max_ball_speed, 1.0F), 0.0F, 1.0F);
  return toward_goal * delta_frac * dense_cfg_.touch_direction_weight;
}

float RewardEngine::speed_toward_ball(
    const CarState& car, const EnvState& env) const {
  if (dense_cfg_.speed_toward_ball_weight <= 0.0F) return 0.0F;

  const Vec3 to_ball{
      env.ball.position.x - car.position.x,
      env.ball.position.y - car.position.y,
      env.ball.position.z - car.position.z,
  };
  const float dist = vec3_magnitude(to_ball);
  if (dist < 1.0F) return 0.0F;

  const Vec3 to_ball_norm{
      to_ball.x / dist,
      to_ball.y / dist,
      to_ball.z / dist,
  };

  const float speed_toward = vec3_dot(car.velocity, to_ball_norm);
  const float car_speed = vec3_magnitude(car.velocity);
  const float frac = clamp(speed_toward / std::max(car_speed, 1.0F), 0.0F, 1.0F);
  const float decay = std::exp(-dist / dense_cfg_.speed_toward_ball_decay);

  return frac * decay * dense_cfg_.speed_toward_ball_weight;
}

float RewardEngine::face_ball(
    const CarState& car, const EnvState& env) const {
  if (dense_cfg_.face_ball_weight <= 0.0F) return 0.0F;

  const Vec3 to_ball{
      env.ball.position.x - car.position.x,
      env.ball.position.y - car.position.y,
      env.ball.position.z - car.position.z,
  };
  const float dist = vec3_magnitude(to_ball);
  if (dist < 1.0F) return dense_cfg_.face_ball_weight;

  const Vec3 to_ball_norm{
      to_ball.x / dist,
      to_ball.y / dist,
      to_ball.z / dist,
  };

  const float facing = vec3_dot(car.forward, to_ball_norm);
  const float decay = std::exp(-dist / dense_cfg_.speed_toward_ball_decay);
  return std::max(0.0F, facing) * decay * dense_cfg_.face_ball_weight;
}

float RewardEngine::air_reward(
    const CarState& car, const EnvState& env) const {
  (void)env;
  if (dense_cfg_.air_reward_weight <= 0.0F) return 0.0F;
  if (car.on_ground) return 0.0F;

  return dense_cfg_.air_reward_weight;
}

float RewardEngine::velocity_ball_to_goal(
    const CarState& car, const EnvState& env) const {
  if (dense_cfg_.velocity_ball_to_goal_weight <= 0.0F) return 0.0F;

  const float team_sign = (car.team == Team::Blue) ? 1.0F : -1.0F;
  const float vel_toward = env.ball.velocity.y * team_sign;
  const float frac = clamp(vel_toward / std::max(dense_cfg_.max_ball_speed, 1.0F), -1.0F, 1.0F);

  return frac * dense_cfg_.velocity_ball_to_goal_weight;
}

float RewardEngine::air_touch(
    const CarState& car, const EnvState& env, float vel_delta_frac) const {
  if (dense_cfg_.air_touch_weight <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;
  if (car.on_ground) return 0.0F;

  const float height_frac = clamp(env.ball.position.z / kCeilingZ, 0.0F, 1.0F);
  const float air_time_frac = clamp(
      car.air_time_since_jump / dense_cfg_.air_touch_max_air_time, 0.0F, 1.0F);
  const float quality = std::min(height_frac, air_time_frac);

  return quality * vel_delta_frac * dense_cfg_.air_touch_weight;
}

float RewardEngine::save_boost(const CarState& car) const {
  if (dense_cfg_.save_boost_weight <= 0.0F) return 0.0F;

  return std::sqrt(std::max(0.0F, car.boost)) * dense_cfg_.save_boost_weight;
}

float RewardEngine::boost_efficiency(const CarState& car, const EnvState& env) const {
  if (dense_cfg_.boost_efficiency_weight <= 0.0F) return 0.0F;

  const float dx = env.ball.position.x - car.position.x;
  const float dy = env.ball.position.y - car.position.y;
  const float dz = env.ball.position.z - car.position.z;
  const float dist_to_ball = std::sqrt(dx * dx + dy * dy + dz * dz);
  const float car_speed = vec3_magnitude(car.velocity);

  const float boost_amt = std::sqrt(std::max(0.0F, car.boost));
  const bool is_engaged = dist_to_ball < 1000.0F || car_speed > 1000.0F;

  // reward having boost when engaged in play
  if (is_engaged) {
    return boost_amt * dense_cfg_.boost_efficiency_weight;
  }

  // small penalty for hoarding boost while idle
  if (car_speed < 200.0F && dist_to_ball > 2000.0F && car.boost > 0.5F) {
    return -boost_amt * dense_cfg_.boost_efficiency_weight * 0.1F;
  }

  return 0.0F;
}

float RewardEngine::boost_used(const CarState& car, AgentRewardState& s) const {
  if (dense_cfg_.boost_used_weight <= 0.0F) return 0.0F;
  if (!car.is_boosting) return 0.0F;

  const float boost_delta = s.prev_boost - car.boost;
  if (boost_delta <= 0.0F) return 0.0F;

  return clamp(boost_delta / 33.0F, 0.0F, 1.0F) * dense_cfg_.boost_used_weight;
}

float RewardEngine::defensive_positioning(const CarState& car, const EnvState& env) const {
  if (dense_cfg_.defensive_positioning_weight <= 0.0F) return 0.0F;

  const float goal_dir = (car.team == Team::Blue) ? 1.0F : -1.0F;
  const float own_goal_y = -goal_dir * kArenaExtentY;

  const float dx = env.ball.position.x - car.position.x;
  const float dy = env.ball.position.y - car.position.y;
  const float dz = env.ball.position.z - car.position.z;
  const float dist_to_ball = std::sqrt(dx * dx + dy * dy + dz * dz);

  if (dist_to_ball < 1.0F) return 0.0F;

  const Vec3 to_ball{dx, dy, dz};
  const float d = std::max(dist_to_ball, 1.0F);
  const Vec3 to_ball_norm{to_ball.x / d, to_ball.y / d, to_ball.z / d};

  const Vec3 to_own_goal{0.0F, own_goal_y, 0.0F};
  const float og_d = std::sqrt(to_own_goal.x * to_own_goal.x + to_own_goal.y * to_own_goal.y + to_own_goal.z * to_own_goal.z);
  if (og_d < 1.0F) return 0.0F;
  const Vec3 to_own_goal_norm{to_own_goal.x / og_d, to_own_goal.y / og_d, to_own_goal.z / og_d};

  // is opponent closer to ball than this agent?
  bool opponent_has_ball = false;
  float closest_opponent_dist = 1.0e9F;
  for (const auto& other : env.cars) {
    if (other.team == car.team) continue;
    const float ox = env.ball.position.x - other.position.x;
    const float oy = env.ball.position.y - other.position.y;
    const float oz = env.ball.position.z - other.position.z;
    const float od = std::sqrt(ox * ox + oy * oy + oz * oz);
    if (od < dist_to_ball) {
      opponent_has_ball = true;
    }
    closest_opponent_dist = std::min(closest_opponent_dist, od);
  }

  if (!opponent_has_ball) return 0.0F;

  const float facing_own_goal = vec3_dot(car.forward, to_own_goal_norm);
  const float exp_dist = std::exp(-dist_to_ball / dense_cfg_.defensive_positioning_decay);

  // reward being between ball and own goal AND actively facing it.
  // Multiplying by max(0, facing_own_goal) requires the car to have rotated
  // to face its goal — a passive defender who never turns to face goal (e.g.
  // one that yielded possession and is just standing there) gets zero reward.
  const float ball_goal_dot = vec3_dot({-to_ball_norm.x, -to_ball_norm.y, -to_ball_norm.z}, to_own_goal_norm);
  return std::max(0.0F, ball_goal_dot) * std::max(0.0F, facing_own_goal) * exp_dist * dense_cfg_.defensive_positioning_weight;
}

float RewardEngine::shot_accuracy(const CarState& car, const EnvState& env) const {
  if (dense_cfg_.shot_accuracy_weight <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;

  const float goal_dir = (car.team == Team::Blue) ? 1.0F : -1.0F;

  // only consider shots from offensive half
  const float car_y_sign = car.position.y * goal_dir;
  if (car_y_sign < 0.0F) return 0.0F;

  const float ball_speed = vec3_magnitude(env.ball.velocity);
  if (ball_speed < 100.0F) return 0.0F;

  const Vec3 ball_vel_norm{
      env.ball.velocity.x / ball_speed,
      env.ball.velocity.y / ball_speed,
      env.ball.velocity.z / ball_speed,
  };
  const float toward_goal = ball_vel_norm.y * goal_dir;

  // Threshold lowered from 0.7 to 0.5 (was ~45-degree cone, now ~60-degree).
  // The 0.7 cliff was too conservative — the bot would only shoot when nearly
  // perfectly aligned with goal, ignoring many viable shooting angles.
  if (toward_goal < 0.5F) return 0.0F;

  const float speed_frac = clamp(ball_speed / std::max(dense_cfg_.max_ball_speed, 1.0F), 0.0F, 1.0F);
  return toward_goal * speed_frac * dense_cfg_.shot_accuracy_weight;
}

float RewardEngine::boost_pickup(
    const CarState& car, AgentRewardState& s) const {
  if (dense_cfg_.boost_pickup_big_weight <= 0.0F
      && dense_cfg_.boost_pickup_small_weight <= 0.0F) {
    return 0.0F;
  }

  const float amount = car.boost - s.prev_boost;
  if (amount <= 0.001F) return 0.0F;

  if (amount > dense_cfg_.boost_pickup_big_threshold) {
    return amount * dense_cfg_.boost_pickup_big_weight;
  }
  return amount * dense_cfg_.boost_pickup_small_weight;
}

float RewardEngine::possession_chain(
    const CarState& car, const EnvState& env, int global_tick, AgentRewardState& s) const {
  const bool enabled = dense_cfg_.possession_chain_weight > 0.0F ||
      dense_cfg_.possession_chain_scale > 0.0F ||
      dense_cfg_.possession_proximity_weight > 0.0F ||
      dense_cfg_.possession_speed_toward_ball_weight > 0.0F ||
      dense_cfg_.possession_face_ball_weight > 0.0F;
  if (!enabled) return 0.0F;

  // Timeout reset: if too much time passed since this agent last touched, break the chain.
  if (s.consecutive_touches > 0 &&
      (global_tick - s.last_own_touch_tick) >= dense_cfg_.possession_chain_timeout_ticks) {
    s.consecutive_touches = 0;
  }

  // Opponent reset: if another agent was or is the env's last toucher, the
  // possession chain is broken before this agent can start a new chain.
  if (s.consecutive_touches > 0 &&
      ((s.prev_env_last_touch_agent >= 0 && s.prev_env_last_touch_agent != car.id) ||
       (env.last_touch_agent >= 0 && env.last_touch_agent != car.id))) {
    s.consecutive_touches = 0;
  }

  float reward = 0.0F;

  // Edge-detect a new touch by this agent.
  if (car.ball_touched && !s.prev_ball_touched) {
    s.last_own_touch_tick = global_tick;
    s.consecutive_touches++;

    reward += dense_cfg_.possession_chain_weight +
              dense_cfg_.possession_chain_scale * static_cast<float>(s.consecutive_touches - 1);
  }

  const bool possession_active = s.consecutive_touches > 0 &&
      env.last_touch_agent == car.id &&
      (global_tick - s.last_own_touch_tick) <= dense_cfg_.possession_window_ticks;
  if (!possession_active) return reward;

  const Vec3 to_ball{
      env.ball.position.x - car.position.x,
      env.ball.position.y - car.position.y,
      env.ball.position.z - car.position.z,
  };
  const float dist = vec3_magnitude(to_ball);
  if (dist < 1.0F) {
    reward += dense_cfg_.possession_proximity_weight;
    return reward;
  }

  const float decay = std::exp(-dist / std::max(dense_cfg_.possession_distance_decay, 1.0F));
  const Vec3 to_ball_norm{
      to_ball.x / dist,
      to_ball.y / dist,
      to_ball.z / dist,
  };

  reward += decay * dense_cfg_.possession_proximity_weight;

  const float car_speed = vec3_magnitude(car.velocity);
  const float speed_toward = vec3_dot(car.velocity, to_ball_norm);
  const float speed_frac = clamp(speed_toward / std::max(car_speed, 1.0F), 0.0F, 1.0F);
  reward += speed_frac * decay * dense_cfg_.possession_speed_toward_ball_weight;

  const float facing = std::max(0.0F, vec3_dot(car.forward, to_ball_norm));
  reward += facing * decay * dense_cfg_.possession_face_ball_weight;

  return reward;
}

// --- mechanic reward methods ---------------------------------------------------

float RewardEngine::detect_speed_flip(const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("speed_flip")) return 0.0F;
  if (mechanic_cfg_.speed_flip <= 0.0F) return 0.0F;
  // Speed flip: flip starts from ground, car stays low and boosts.
  // Requiring prev_on_ground and low altitude filters out aerial flips.
  if (s.prev_on_ground && !s.prev_is_flipping && car.is_flipping && car.is_boosting) {
    const float z_above_ground = car.position.z - kGroundZ;
    if (z_above_ground < 150.0F) {
      return mechanic_cfg_.speed_flip;
    }
  }
  return 0.0F;
}

float RewardEngine::detect_wavedash(int tick, const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("wavedash")) return 0.0F;
  if (mechanic_cfg_.wavedash <= 0.0F) return 0.0F;
  // Wavedash: car was flipping close to ground, then lands (ground cancels
  // the flip, converting flip momentum into forward speed).
  // Detecting on landing prevents exploiting by just jumping + flipping near
  // the ground without actually making ground contact.
  if (car.on_ground && !s.prev_on_ground && s.prev_is_flipping) {
    const float prev_z_above_ground = s.prev_car_position_z - kGroundZ;
    if (prev_z_above_ground < kWavedashZThreshold) {
      s.last_wavedash_tick = tick;
      return mechanic_cfg_.wavedash;
    }
  }
  return 0.0F;
}

float RewardEngine::detect_chain_dash(int tick, int prev_dash_tick, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("chain_dash")) return 0.0F;
  (void)s;
  if (mechanic_cfg_.chain_dash_bonus <= 0.0F) return 0.0F;
  if (prev_dash_tick >= 0 && (tick - prev_dash_tick) <= kChainDashWindowTicks &&
      (tick - prev_dash_tick) > 0) {
    return mechanic_cfg_.chain_dash_bonus;
  }
  return 0.0F;
}

float RewardEngine::detect_half_flip(const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("half_flip")) return 0.0F;
  if (mechanic_cfg_.half_flip <= 0.0F) return 0.0F;
  if (!car.on_ground && !s.prev_is_flipping && car.is_flipping && !car.has_double_jumped) {
    const float vel_mag = vec3_magnitude(car.velocity);
    if (vel_mag < 10.0F) return 0.0F;
    const Vec3 vel_norm = {
        car.velocity.x / vel_mag,
        car.velocity.y / vel_mag,
        car.velocity.z / vel_mag,
    };
    const float fwd_dot_vel = vec3_dot(car.forward, vel_norm);
    // car.handbrake is the ANALOG powerslide value (handbrakeVal), which rises
    // at POWERSLIDE_RISE_RATE = 5/s → ~0.042 per tick at 120 Hz.  The old
    // threshold of 0.5 required ~12 continuous ticks of powerslide BEFORE the
    // flip, making this reward essentially unreachable.  Use 0.01 instead,
    // which is satisfied after just one tick of handbrake input.
    if (fwd_dot_vel < -0.3F && car.handbrake > 0.01F) {
      return mechanic_cfg_.half_flip;
    }
  }
  return 0.0F;
}

float RewardEngine::detect_wall_dash(int tick, const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("wall_dash")) return 0.0F;
  if (mechanic_cfg_.wall_dash <= 0.0F) return 0.0F;
  // Same landing-gated detection as wavedash, but requires previous position
  // to be near an arena wall.
  if (car.on_ground && !s.prev_on_ground && s.prev_is_flipping) {
    const float prev_z_above_ground = s.prev_car_position_z - kGroundZ;
    if (prev_z_above_ground < kWavedashZThreshold) {
      const bool near_wall =
          std::abs(car.position.x) > (kArenaExtentX - kWallDashDist) ||
          std::abs(car.position.y) > (kArenaExtentY - kWallDashDist);
      if (near_wall) {
        s.last_wavedash_tick = tick;
        return mechanic_cfg_.wall_dash;
      }
    }
  }
  return 0.0F;
}

float RewardEngine::detect_air_dribble(const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("air_dribble")) return 0.0F;
  if (mechanic_cfg_.air_dribble_base <= 0.0F) return 0.0F;

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
    return mechanic_cfg_.air_dribble_base + mechanic_cfg_.air_dribble_scale * scale;
  }

  return 0.0F;
}

float RewardEngine::detect_flip_reset(const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("flip_reset")) return 0.0F;
  if (mechanic_cfg_.flip_reset <= 0.0F) return 0.0F;
  if (!s.prev_has_flip_reset && car.has_flip_reset) {
    return mechanic_cfg_.flip_reset;
  }
  return 0.0F;
}

float RewardEngine::detect_ceiling_shot(const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("ceiling_shot")) return 0.0F;
  if (mechanic_cfg_.ceiling_shot <= 0.0F) return 0.0F;

  // In RocketSim, isOnGround = (numWheelsInContact >= 3), which is TRUE when
  // the car is DRIVING ON THE CEILING as well as the floor.  The original
  // check "!car.on_ground && z > kCeilingThreshold" therefore never fired
  // while the car was actually on the ceiling surface, and the reset
  // "if (car.on_ground)" cleared the flag the moment the car touched the
  // ceiling, making ceiling shots from ceiling rides impossible to detect.
  //
  // Fix: set the flag whenever z is above the threshold (regardless of ground
  // state), and reset it only when on a FLOOR surface (z < kCeilingThreshold).
  if (car.position.z > kCeilingThreshold) {
    s.was_on_ceiling = true;
  }

  if (s.was_on_ceiling && car.position.z < kCeilingThreshold - 100.0F) {
    if (car.ball_touched && !car.on_ground) {
      s.was_on_ceiling = false;
      return mechanic_cfg_.ceiling_shot;
    }
  }

  // Only reset when landing on the floor, not the ceiling surface.
  if (car.on_ground && car.position.z < kCeilingThreshold) {
    s.was_on_ceiling = false;
  }

  return 0.0F;
}

float RewardEngine::detect_double_tap(int tick, const CarState& car, const EnvState& env,
                                       AgentRewardState& s) const {
  if (!is_mechanic_unlocked("double_tap")) return 0.0F;
  if (mechanic_cfg_.double_tap <= 0.0F) return 0.0F;

  // improved backboard detection: ball must be near back wall.
  // Only set the flag — never unconditionally clear it so it persists
  // after the ball bounces back away from the board.
  const float abs_y = std::abs(env.ball.position.y);
  if (abs_y > 5000.0F) {
    s.ball_hit_backboard = true;
  }

  // Check for the second (air) touch BEFORE updating last_ball_touch_tick so
  // that (tick - last_ball_touch_tick) > 0 can actually be true.
  if (car.ball_touched && s.ball_hit_backboard && s.last_toucher_id == car.id &&
      !car.on_ground &&
      (tick - s.last_ball_touch_tick) <= kDoubleTapWindowTicks &&
      (tick - s.last_ball_touch_tick) > 0) {
    s.ball_hit_backboard = false;
    s.last_ball_touch_tick = tick;
    return mechanic_cfg_.double_tap;
  }

  // Record the touch AFTER the detection check so the first touch sets the
  // baseline tick and the next touch can measure the elapsed time.
  if (car.ball_touched) {
    s.last_ball_touch_tick = tick;
    s.last_toucher_id = car.id;
  }

  return 0.0F;
}

float RewardEngine::detect_preflip(const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("preflip")) return 0.0F;
  if (mechanic_cfg_.preflip <= 0.0F) return 0.0F;

  if (!s.prev_is_flipping && car.is_flipping) {
    s.flip_start_tick = 1;
  }

  if (car.is_flipping && car.ball_touched && !s.prev_ball_touched) {
    if (s.flip_start_tick > 0) {
      s.flip_start_tick = -1;
      return mechanic_cfg_.preflip;
    }
  }

  if (!car.is_flipping) {
    s.flip_start_tick = -1;
  }

  return 0.0F;
}

float RewardEngine::detect_redirect(const CarState& car, const EnvState& env,
                                     AgentRewardState& s, Team team) const {
  if (!is_mechanic_unlocked("redirect")) return 0.0F;
  if (mechanic_cfg_.redirect <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;

  const Vec3 prev_vel = s.prev_ball_velocity;
  const Vec3 post_vel = env.ball.velocity;
  const float prev_mag = vec3_magnitude(prev_vel);
  const float post_mag = vec3_magnitude(post_vel);

  if (prev_mag < 100.0F || post_mag < 100.0F) return 0.0F;

  // compute angle between prev and post direction
  const float dot = clamp((prev_vel.x * post_vel.x + prev_vel.y * post_vel.y + prev_vel.z * post_vel.z) / (prev_mag * post_mag), -1.0F, 1.0F);
  const float angle = std::acos(dot);
  const float angle_deg = angle * (180.0F / 3.14159265358979323846F);

  if (angle_deg < 30.0F) return 0.0F;

  const float goal_dir = (team == Team::Blue) ? 1.0F : -1.0F;
  const float post_y = post_vel.y / post_mag;

  // stricter: must redirect ball toward goal
  if (post_y * goal_dir > 0.3F) {
    return mechanic_cfg_.redirect;
  }

  return 0.0F;
}

float RewardEngine::detect_pogo(const CarState& car, AgentRewardState& s) const {
  if (!is_mechanic_unlocked("pogo")) return 0.0F;
  if (mechanic_cfg_.pogo <= 0.0F) return 0.0F;

  if (!s.prev_on_ground && car.on_ground) {
    if (!car.is_jumping) {
      // Use the z-velocity from the PREVIOUS tick (before the landing
      // collision zeroes it out).  s.prev_car_velocity_z is saved in
      // compute() after all reward functions run.
      if (s.prev_car_velocity_z > 200.0F) {
        return mechanic_cfg_.pogo;
      }
    }
  }

  return 0.0F;
}

float RewardEngine::detect_pinch(const CarState& car, const EnvState& env,
                                  AgentRewardState& s) const {
  if (!is_mechanic_unlocked("pinch")) return 0.0F;
  if (mechanic_cfg_.pinch <= 0.0F) return 0.0F;

  // causality: agent must be near the ball
  const float dx = car.position.x - env.ball.position.x;
  const float dy = car.position.y - env.ball.position.y;
  const float dz = car.position.z - env.ball.position.z;
  const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (dist > 250.0F) return 0.0F;

  const float prev_mag = vec3_magnitude(s.prev_ball_velocity);
  const float post_mag = vec3_magnitude(env.ball.velocity);
  const float delta = post_mag - prev_mag;

  if (delta > kPinchVelocityDelta) {
    // verify velocity change points roughly from agent toward ball
    const Vec3 vel_delta{
        env.ball.velocity.x - s.prev_ball_velocity.x,
        env.ball.velocity.y - s.prev_ball_velocity.y,
        env.ball.velocity.z - s.prev_ball_velocity.z,
    };
    const float vd_mag = vec3_magnitude(vel_delta);
    if (vd_mag > 0.01F) {
      const float align = vec3_dot(vel_delta, { -dx, -dy, -dz }) / (vd_mag * dist);
      if (align > 0.3F) {
        return mechanic_cfg_.pinch;
      }
    }
  }

  return 0.0F;
}

float RewardEngine::detect_team_pinch(const CarState& car, const EnvState& env,
                                       AgentRewardState& s, int env_team_size) const {
  (void)env_team_size;
  if (!is_mechanic_unlocked("team_pinch")) return 0.0F;
  if (mechanic_cfg_.team_pinch <= 0.0F) return 0.0F;

  // causality: agent must be near the ball
  const float dx = car.position.x - env.ball.position.x;
  const float dy = car.position.y - env.ball.position.y;
  const float dz = car.position.z - env.ball.position.z;
  const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (dist > 250.0F) return 0.0F;

  const float prev_mag = vec3_magnitude(s.prev_ball_velocity);
  const float post_mag = vec3_magnitude(env.ball.velocity);
  const float delta = post_mag - prev_mag;
  if (delta < kPinchVelocityDelta) return 0.0F;

  int teammate_near_ball = 0;
  for (const auto& other_car : env.cars) {
    if (other_car.id == car.id) continue;
    if (other_car.team != car.team) continue;
    const float ox = other_car.position.x - env.ball.position.x;
    const float oy = other_car.position.y - env.ball.position.y;
    const float oz = other_car.position.z - env.ball.position.z;
    const float od = std::sqrt(ox * ox + oy * oy + oz * oz);
    if (od < 350.0F) {
      teammate_near_ball++;
    }
  }

  if (teammate_near_ball >= 1) {
    return mechanic_cfg_.team_pinch;
  }

  return 0.0F;
}

float RewardEngine::detect_kickoff_first_touch(const CarState& car, const EnvState& env,
                                                EnvRewardState& env_state) const {
  if (!is_mechanic_unlocked("kickoff_first_touch")) return 0.0F;
  if (mechanic_cfg_.kickoff_first_touch <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;
  if (env_state.kickoff_reward_given) return 0.0F;

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
    env_state.first_touch_team = 2;
    // 50/50 — do not award kickoff_first_touch; kickoff_50_50_touch handles this.
    return 0.0F;
  }

  env_state.first_touch_team = team_int;
  env_state.kickoff_reward_given = true;
  return mechanic_cfg_.kickoff_first_touch;
}

float RewardEngine::detect_kickoff_50_50_touch(const CarState& car, const EnvState& env,
                                                EnvRewardState& env_state) const {
  if (!is_mechanic_unlocked("kickoff_50_50_touch")) return 0.0F;
  if (mechanic_cfg_.kickoff_50_50_touch <= 0.0F) return 0.0F;
  if (!car.ball_touched) return 0.0F;
  // No kickoff_reward_given gate — each agent in a 50/50 earns this independently.
  // The check below (both teams touching simultaneously) is self-limiting.

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
    env_state.kickoff_reward_given = true;
    return mechanic_cfg_.kickoff_50_50_touch;
  }

  return 0.0F;
}

}  // namespace pulsar
