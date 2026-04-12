#include "pulsar/env/rocketsim_engine.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>

#ifdef PULSAR_HAS_ROCKETSIM
#include "RocketSim.h"
#endif

namespace pulsar {
namespace {

std::filesystem::path resolve_collision_meshes_path(const std::string& configured_path) {
  namespace fs = std::filesystem;
  fs::path path(configured_path);
  if (path.is_absolute() && fs::exists(path)) {
    return path;
  }

  if (fs::exists(path)) {
    return fs::absolute(path);
  }

  fs::path current = fs::current_path();
  for (int depth = 0; depth < 8; ++depth) {
    const fs::path candidate = current / path;
    if (fs::exists(candidate)) {
      return fs::absolute(candidate);
    }
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }

  return path;
}

float team_sign(Team team) {
  return team == Team::Blue ? 1.0F : -1.0F;
}

float magnitude(const Vec3& value) {
  return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

Vec3 normalize(const Vec3& value) {
  const float len = magnitude(value);
  if (len < 1.0e-6F) {
    return {1.0F, 0.0F, 0.0F};
  }
  return value * (1.0F / len);
}

#ifdef PULSAR_HAS_ROCKETSIM
RocketSim::Team to_rs_team(Team team) {
  return team == Team::Blue ? RocketSim::Team::BLUE : RocketSim::Team::ORANGE;
}

Vec3 from_rs_vec(const RocketSim::Vec& value) {
  return {value.x, value.y, value.z};
}
#endif

}  // namespace

RocketSimTransitionEngine::RocketSimTransitionEngine(EnvConfig config, StateMutatorPtr reset_mutator)
    : config_(std::move(config)), reset_mutator_(std::move(reset_mutator)) {
  batched_state_.num_envs = 1;
  batched_state_.agents_per_env = static_cast<std::size_t>(config_.team_size * 2);
  reset(config_.seed);
}

RocketSimTransitionEngine::~RocketSimTransitionEngine() {
#ifdef PULSAR_HAS_ROCKETSIM
  delete arena_;
  arena_ = nullptr;
#endif
}

void RocketSimTransitionEngine::reset(std::uint64_t seed) {
#ifdef PULSAR_HAS_ROCKETSIM
  static std::once_flag rocketsim_init_once;
  std::call_once(rocketsim_init_once, [this]() {
    RocketSim::Init(resolve_collision_meshes_path(config_.collision_meshes_path), true);
  });

  delete arena_;
  arena_ = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
  if (arena_ == nullptr) {
    throw std::runtime_error("RocketSim arena creation failed.");
  }

  cars_.clear();
  cars_.reserve(static_cast<std::size_t>(config_.team_size * 2));
  for (int i = 0; i < config_.team_size; ++i) {
    cars_.push_back(arena_->AddCar(RocketSim::Team::BLUE));
  }
  for (int i = 0; i < config_.team_size; ++i) {
    cars_.push_back(arena_->AddCar(RocketSim::Team::ORANGE));
  }

  state_ = {};
  state_.cars.resize(cars_.size());
  state_.boost_pad_timers.resize(arena_->GetBoostPads().size(), 0.0F);
  state_.goal_scored = false;
  state_.last_touch_tick = 0;
  episode_ticks_ = 0;
  last_reset_seed_ = seed;

  arena_->SetGoalScoreCallback(
      [](RocketSim::Arena*, RocketSim::Team scoring_team, void* user_info) {
        auto* self = static_cast<RocketSimTransitionEngine*>(user_info);
        self->state_.goal_scored = true;
        if (scoring_team == RocketSim::Team::BLUE) {
          self->state_.blue_score += 1;
        } else {
          self->state_.orange_score += 1;
        }
      },
      this);

  const int kickoff_seed = config_.randomize_kickoffs ? static_cast<int>(seed) : 0;
  arena_->ResetToRandomKickoff(kickoff_seed);
  sync_batched_state();
#else
  state_ = {};
  episode_ticks_ = 0;

  if (reset_mutator_) {
    reset_mutator_->apply(state_, seed);
  }

  sync_batched_state();
#endif
}

StepResult RocketSimTransitionEngine::step(std::span<const ControllerState> actions) {
  step_inplace(actions);

  StepResult result;
  result.state = state_;
  result.rewards.assign(state_.cars.size(), 0.0F);
  result.terminated.assign(state_.cars.size(), 0);
  result.truncated.assign(state_.cars.size(), 0);
  return result;
}

void RocketSimTransitionEngine::step_inplace(std::span<const ControllerState> actions) {
  if (actions.size() != state_.cars.size()) {
    throw std::invalid_argument("Action count must match the number of cars.");
  }

#ifdef PULSAR_HAS_ROCKETSIM
  if (arena_ == nullptr) {
    throw std::runtime_error("RocketSim arena is not initialized.");
  }

  state_.goal_scored = false;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    RocketSim::CarControls controls;
    controls.throttle = actions[i].throttle;
    controls.steer = actions[i].steer;
    controls.yaw = actions[i].yaw;
    controls.pitch = actions[i].pitch;
    controls.roll = actions[i].roll;
    controls.jump = actions[i].jump;
    controls.boost = actions[i].boost;
    controls.handbrake = actions[i].handbrake;
    controls.ClampFix();
    cars_[i]->controls = controls;
  }

  arena_->Step(config_.tick_skip);
  episode_ticks_ += config_.tick_skip;
  sync_batched_state();
#else
  apply_placeholder_dynamics(actions);
  state_.tick += config_.tick_skip;
  episode_ticks_ += config_.tick_skip;
  sync_batched_state();
#endif
}

const EnvState& RocketSimTransitionEngine::state() const {
  return state_;
}

std::size_t RocketSimTransitionEngine::num_agents() const {
  return state_.cars.size();
}

const BatchedArenaState& RocketSimTransitionEngine::batched_state() const {
  return batched_state_;
}

void RocketSimTransitionEngine::sync_batched_state() {
#ifdef PULSAR_HAS_ROCKETSIM
  if (arena_ != nullptr) {
    const auto ball_state = arena_->ball->GetState();
    state_.tick = static_cast<int>(arena_->tickCount);
    state_.ball.position = from_rs_vec(ball_state.pos);
    state_.ball.velocity = from_rs_vec(ball_state.vel);
    state_.ball.angular_velocity = from_rs_vec(ball_state.angVel);
    if (state_.cars.size() != cars_.size()) {
      state_.cars.resize(cars_.size());
    }

    int max_touch_tick = state_.last_touch_tick;
    int last_touch_agent = state_.last_touch_agent;

    for (std::size_t i = 0; i < cars_.size(); ++i) {
      auto* car_ptr = cars_[i];
      const auto car_state = car_ptr->GetState();
      CarState& car = state_.cars[i];
      car.id = static_cast<int>(car_ptr->id);
      car.team = car_ptr->team == RocketSim::Team::BLUE ? Team::Blue : Team::Orange;
      car.position = from_rs_vec(car_state.pos);
      car.velocity = from_rs_vec(car_state.vel);
      car.angular_velocity = from_rs_vec(car_state.angVel);
      car.forward = from_rs_vec(car_state.rotMat.forward);
      car.up = from_rs_vec(car_state.rotMat.up);
      car.boost = car_state.boost;
      car.on_ground = car_state.isOnGround;
      car.has_flip = car_state.HasFlipOrJump();
      car.ball_touched = false;
      car.is_holding_jump = car_state.lastControls.jump;
      car.has_jumped = car_state.hasJumped;
      car.is_jumping = car_state.isJumping;
      car.has_flipped = car_state.hasFlipped;
      car.is_flipping = car_state.isFlipping;
      car.has_double_jumped = car_state.hasDoubleJumped;
      car.is_boosting = car_state.isBoosting;
      car.is_supersonic = car_state.isSupersonic;
      car.air_time_since_jump = car_state.airTimeSinceJump;
      car.handbrake = car_state.handbrakeVal;
      car.demo_respawn_timer = car_state.demoRespawnTimer;

      if (car_state.ballHitInfo.isValid) {
        const int touch_tick = static_cast<int>(car_state.ballHitInfo.tickCountWhenHit);
        if (touch_tick >= max_touch_tick) {
          max_touch_tick = touch_tick;
          last_touch_agent = car.id;
        }
        car.ball_touched =
            (static_cast<int>(arena_->tickCount) - touch_tick) <= config_.tick_skip;
      } else {
        car.ball_touched = false;
      }
    }

    state_.last_touch_tick = max_touch_tick;
    state_.last_touch_agent = last_touch_agent;
    state_.kickoff_pause = false;

    if (state_.boost_pad_timers.size() != arena_->GetBoostPads().size()) {
      state_.boost_pad_timers.resize(arena_->GetBoostPads().size());
    }
    for (std::size_t i = 0; i < arena_->GetBoostPads().size(); ++i) {
      const auto* pad = arena_->GetBoostPads()[i];
      const auto pad_state = pad->GetState();
      state_.boost_pad_timers[i] = pad_state.isActive ? 0.0F : pad_state.cooldown;
    }
  }
#endif

  batched_state_.ball_positions.resize(3);
  batched_state_.ball_velocities.resize(3);
  batched_state_.ball_positions[0] = state_.ball.position.x;
  batched_state_.ball_positions[1] = state_.ball.position.y;
  batched_state_.ball_positions[2] = state_.ball.position.z;
  batched_state_.ball_velocities[0] = state_.ball.velocity.x;
  batched_state_.ball_velocities[1] = state_.ball.velocity.y;
  batched_state_.ball_velocities[2] = state_.ball.velocity.z;

  if (batched_state_.car_positions.size() != state_.cars.size() * 3) {
    batched_state_.car_positions.resize(state_.cars.size() * 3);
    batched_state_.car_velocities.resize(state_.cars.size() * 3);
    batched_state_.car_boost.resize(state_.cars.size());
  }

  for (std::size_t i = 0; i < state_.cars.size(); ++i) {
    const auto& car = state_.cars[i];
    const std::size_t base = i * 3;
    batched_state_.car_positions[base] = car.position.x;
    batched_state_.car_positions[base + 1] = car.position.y;
    batched_state_.car_positions[base + 2] = car.position.z;
    batched_state_.car_velocities[base] = car.velocity.x;
    batched_state_.car_velocities[base + 1] = car.velocity.y;
    batched_state_.car_velocities[base + 2] = car.velocity.z;
    batched_state_.car_boost[i] = car.boost;
  }
}

void RocketSimTransitionEngine::apply_placeholder_dynamics(std::span<const ControllerState> actions) {
  static constexpr float kDt = 8.0F / 120.0F;
  static constexpr float kAccel = 900.0F;

  state_.goal_scored = false;

  for (std::size_t i = 0; i < state_.cars.size(); ++i) {
    auto& car = state_.cars[i];
    const auto& action = actions[i];

    car.velocity.x += action.steer * 120.0F * kDt;
    car.velocity.y += team_sign(car.team) * action.throttle * kAccel * kDt;
    car.position = car.position + car.velocity * kDt;
    car.boost = std::clamp(car.boost - (action.boost ? 1.0F : 0.0F), 0.0F, 100.0F);
    car.ball_touched = false;
    car.is_boosting = action.boost;
    car.is_supersonic = std::sqrt(car.velocity.x * car.velocity.x + car.velocity.y * car.velocity.y) > 2200.0F;
    car.handbrake = action.handbrake ? 1.0F : 0.0F;
    car.forward = normalize(Vec3{action.steer, team_sign(car.team) * std::max(0.1F, action.throttle), 0.0F});
    car.up = {0.0F, 0.0F, 1.0F};

    const Vec3 delta = state_.ball.position - car.position;
    const float dist_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    if (dist_sq < 250.0F * 250.0F) {
      state_.ball.velocity = car.velocity * 1.2F;
      state_.last_touch_agent = car.id;
      state_.last_touch_tick = state_.tick;
      car.ball_touched = true;
      state_.kickoff_pause = false;
    }
  }

  state_.ball.position = state_.ball.position + state_.ball.velocity * kDt;
  state_.ball.velocity = state_.ball.velocity * 0.995F;

  if (state_.ball.position.y > 5120.0F) {
    state_.blue_score += 1;
    state_.goal_scored = true;
  } else if (state_.ball.position.y < -5120.0F) {
    state_.orange_score += 1;
    state_.goal_scored = true;
  }
}

}  // namespace pulsar
