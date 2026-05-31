#include "pulsar/env/mutators.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace pulsar {
namespace {

float lane_offset_for_index(int index) {
  static constexpr float kOffsets[] = {-900.0F, -300.0F, 300.0F, 900.0F};
  return kOffsets[index % 4];
}

}  // namespace

FixedTeamSizeMutator::FixedTeamSizeMutator(EnvConfig config) : config_(std::move(config)) {}

void FixedTeamSizeMutator::apply(EnvState& state, std::uint64_t) const {
  state.cars.clear();
  state.cars.reserve(static_cast<std::size_t>(config_.team_size * 2));
  state.boost_pad_timers.assign(34, 0.0F);

  int next_id = 0;
  for (int team_index = 0; team_index < 2; ++team_index) {
    const Team team = team_index == 0 ? Team::Blue : Team::Orange;
    const float base_y = team == Team::Blue ? -2500.0F : 2500.0F;

    for (int slot = 0; slot < config_.team_size; ++slot) {
      CarState car;
      car.id = next_id++;
      car.team = team;
      car.position = {lane_offset_for_index(slot), base_y, 17.0F};
      car.boost = 33.0F;
      car.forward = {0.0F, team == Team::Blue ? 1.0F : -1.0F, 0.0F};
      state.cars.push_back(car);
    }
  }
}

KickoffMutator::KickoffMutator(EnvConfig config) : config_(std::move(config)) {}

void KickoffMutator::apply(EnvState& state, std::uint64_t) const {
  state.ball.position = {0.0F, 0.0F, 92.75F};
  state.ball.velocity = {0.0F, 0.0F, 0.0F};
  state.ball.angular_velocity = {0.0F, 0.0F, 0.0F};
  state.goal_scored = false;
  state.kickoff_pause = true;
  state.last_touch_agent = -1;
  state.last_touch_tick = 0;
  state.tick = 0;

  for (auto& car : state.cars) {
    car.velocity = {0.0F, 0.0F, 0.0F};
    car.angular_velocity = {0.0F, 0.0F, 0.0F};
    car.ball_touched = false;
    car.is_boosting = false;
    car.is_supersonic = false;
  }
}

RandomStateMutator::RandomStateMutator(EnvConfig config) : config_(std::move(config)) {}

void RandomStateMutator::apply(EnvState& state, std::uint64_t seed) const {
  std::mt19937_64 rng(seed);
  auto rand_range = [&](float min_val, float max_val) -> float {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    return dist(rng);
  };

  // 1. Randomize ball position (within arena boundaries, keeping z off ground)
  state.ball.position.x = rand_range(-3000.0F, 3000.0F);
  state.ball.position.y = rand_range(-4000.0F, 4000.0F);
  state.ball.position.z = rand_range(100.0F, 1200.0F);

  // Randomize ball velocity (up to ~2000 uu/s)
  state.ball.velocity.x = rand_range(-1500.0F, 1500.0F);
  state.ball.velocity.y = rand_range(-1500.0F, 1500.0F);
  state.ball.velocity.z = rand_range(-800.0F, 800.0F);

  // Randomize ball angular velocity
  state.ball.angular_velocity.x = rand_range(-3.0F, 3.0F);
  state.ball.angular_velocity.y = rand_range(-3.0F, 3.0F);
  state.ball.angular_velocity.z = rand_range(-3.0F, 3.0F);

  state.goal_scored = false;
  state.kickoff_pause = false;
  state.last_touch_agent = -1;
  state.last_touch_tick = 0;
  state.tick = 0;

  // 2. Randomize cars
  for (std::size_t i = 0; i < state.cars.size(); ++i) {
    auto& car = state.cars[i];
    
    // Blue team: y < 0, Orange team: y > 0
    float team_sign = (car.team == Team::Blue) ? -1.0F : 1.0F;
    
    float slot_offset_x = (static_cast<float>(i % 3) - 1.0F) * 800.0F;
    float slot_offset_y = (static_cast<float>(i / 3)) * 600.0F;
    
    car.position.x = std::clamp(rand_range(-2500.0F, 2500.0F) + slot_offset_x, -3500.0F, 3500.0F);
    car.position.y = std::clamp(team_sign * rand_range(1000.0F, 4000.0F) + slot_offset_y * team_sign, -4800.0F, 4800.0F);
    
    // Spawning on ground (z=17) or sometimes in the air (aerial setups, 20% rate)
    if (rand_range(0.0F, 1.0F) < 0.2F) {
      car.position.z = rand_range(100.0F, 800.0F);
      car.velocity.z = rand_range(-300.0F, 500.0F);
      car.on_ground = false;
    } else {
      car.position.z = 17.0F;
      car.velocity.z = 0.0F;
      car.on_ground = true;
    }

    // Contact-seeding: on a fraction of resets, spawn the car right next to the
    // ball (on the ground) so the first few actions touch it. This seeds the
    // buffer with ball-contact transitions — without them the contrastive ball
    // head has no data in which actions move the ball and can never learn.
    constexpr float kContactStartProb = 0.35F;
    constexpr float kTouchDistance = 150.0F;  // ball radius (~92) + car half-length
    if (rand_range(0.0F, 1.0F) < kContactStartProb) {
      const float approach = rand_range(0.0F, 6.2831853F);
      car.position.x = std::clamp(
          state.ball.position.x + kTouchDistance * std::cos(approach), -4000.0F, 4000.0F);
      car.position.y = std::clamp(
          state.ball.position.y + kTouchDistance * std::sin(approach), -4800.0F, 4800.0F);
      car.position.z = 17.0F;
      car.velocity.z = 0.0F;
      car.on_ground = true;
    } else {
      // Otherwise keep a safe distance (min 400 units) to avoid overlapping resets.
      float dx = car.position.x - state.ball.position.x;
      float dy = car.position.y - state.ball.position.y;
      float dz = car.position.z - state.ball.position.z;
      float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 400.0F) {
        car.position.y += team_sign * 500.0F;
        car.position.y = std::clamp(car.position.y, -4800.0F, 4800.0F);
      }
    }

    car.velocity.x = rand_range(-1000.0F, 1000.0F);
    car.velocity.y = rand_range(-1000.0F, 1000.0F);

    car.angular_velocity.x = rand_range(-1.0F, 1.0F);
    car.angular_velocity.y = rand_range(-1.0F, 1.0F);
    car.angular_velocity.z = rand_range(-1.0F, 1.0F);

    // Orientation: point generally towards the ball's position with some noise
    float diff_x = state.ball.position.x - car.position.x;
    float diff_y = state.ball.position.y - car.position.y;
    float yaw = std::atan2(diff_y, diff_x) + rand_range(-0.5F, 0.5F);

    car.forward.x = std::cos(yaw);
    car.forward.y = std::sin(yaw);
    car.forward.z = 0.0F;
    car.up.x = 0.0F;
    car.up.y = 0.0F;
    car.up.z = 1.0F;

    car.boost = rand_range(0.0F, 100.0F);
    car.ball_touched = false;
    car.is_boosting = false;
    car.is_supersonic = false;
    car.has_flip = true;
    car.has_flip_reset = false;
  }
}

MutatorSequence::MutatorSequence(std::vector<StateMutatorPtr> mutators) : mutators_(std::move(mutators)) {}

void MutatorSequence::apply(EnvState& state, std::uint64_t seed) const {
  for (const auto& mutator : mutators_) {
    if (mutator) {
      mutator->apply(state, seed);
    }
  }
}

}  // namespace pulsar
