#include "pulsar/env/done.hpp"

#include <algorithm>
#include <stdexcept>

namespace pulsar {

SimpleDoneCondition::SimpleDoneCondition(EnvConfig config) : config_(std::move(config)) {}

std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>> SimpleDoneCondition::is_done(
    const EnvState& state,
    int episode_ticks) const {
  const std::size_t count = state.cars.size();
  std::vector<std::uint8_t> terminated(count, 0);
  std::vector<std::uint8_t> truncated(count, 0);

  const bool goal_scored = state.goal_scored;
  const bool timeout = episode_ticks >= config_.max_episode_ticks;
  const int no_touch_timeout_ticks =
      static_cast<int>(config_.no_touch_timeout_seconds * static_cast<float>(config_.tick_rate));
  const bool no_touch_timeout = (state.tick - state.last_touch_tick) >= no_touch_timeout_ticks;

  if (goal_scored) {
    std::fill(terminated.begin(), terminated.end(), 1);
  }
  if (timeout || no_touch_timeout) {
    std::fill(truncated.begin(), truncated.end(), 1);
  }

  return {terminated, truncated};
}

void SimpleDoneCondition::is_done_into(
    const EnvState& state,
    int episode_ticks,
    std::span<std::uint8_t> terminated,
    std::span<std::uint8_t> truncated) const {
  if (terminated.size() != state.cars.size() || truncated.size() != state.cars.size()) {
    throw std::invalid_argument("SimpleDoneCondition::is_done_into output spans have incorrect size.");
  }

  std::fill(terminated.begin(), terminated.end(), 0);
  std::fill(truncated.begin(), truncated.end(), 0);

  const bool goal_scored = state.goal_scored;
  const bool timeout = episode_ticks >= config_.max_episode_ticks;
  const int no_touch_timeout_ticks =
      static_cast<int>(config_.no_touch_timeout_seconds * static_cast<float>(config_.tick_rate));
  const bool no_touch_timeout = (state.tick - state.last_touch_tick) >= no_touch_timeout_ticks;

  if (goal_scored) {
    std::fill(terminated.begin(), terminated.end(), 1);
  }
  if (timeout || no_touch_timeout) {
    std::fill(truncated.begin(), truncated.end(), 1);
  }
}

}  // namespace pulsar
