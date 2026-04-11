#include "pulsar/env/mutators.hpp"

#include <algorithm>

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

MutatorSequence::MutatorSequence(std::vector<StateMutatorPtr> mutators) : mutators_(std::move(mutators)) {}

void MutatorSequence::apply(EnvState& state, std::uint64_t seed) const {
  for (const auto& mutator : mutators_) {
    if (mutator) {
      mutator->apply(state, seed);
    }
  }
}

}  // namespace pulsar
