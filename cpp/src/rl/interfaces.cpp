#include "pulsar/rl/interfaces.hpp"

#include <stdexcept>

namespace pulsar {

void TransitionEngine::step_inplace(std::span<const ControllerState> actions) {
  step(actions);
}

void ObsBuilder::build_obs_batch(const EnvState& state, std::span<float> out) const {
  const std::size_t expected = state.cars.size() * obs_dim();
  if (out.size() != expected) {
    throw std::invalid_argument("ObsBuilder::build_obs_batch received an output span with the wrong size.");
  }

  const std::size_t stride = obs_dim();
  for (std::size_t agent_id = 0; agent_id < state.cars.size(); ++agent_id) {
    const std::vector<float> obs = build_obs(state, agent_id);
    std::copy(obs.begin(), obs.end(), out.begin() + static_cast<std::ptrdiff_t>(agent_id * stride));
  }
}

void ActionParser::parse_actions_into(
    std::span<const std::int64_t> action_indices,
    std::span<ControllerState> out) const {
  if (out.size() != action_indices.size()) {
    throw std::invalid_argument("ActionParser::parse_actions_into output span has incorrect size.");
  }

  const std::vector<ControllerState> actions = parse_actions(action_indices);
  std::copy(actions.begin(), actions.end(), out.begin());
}

void ActionParser::build_action_mask_batch(const EnvState& state, std::span<std::uint8_t> out) const {
  if (state.cars.empty()) {
    if (!out.empty()) {
      throw std::invalid_argument("ActionParser::build_action_mask_batch expected an empty output span.");
    }
    return;
  }

  const std::vector<std::uint8_t> first = build_action_mask(state, 0);
  const std::size_t stride = first.size();
  const std::size_t expected = state.cars.size() * stride;
  if (out.size() != expected) {
    throw std::invalid_argument("ActionParser::build_action_mask_batch received an output span with the wrong size.");
  }

  std::copy(first.begin(), first.end(), out.begin());
  for (std::size_t agent_id = 1; agent_id < state.cars.size(); ++agent_id) {
    const std::vector<std::uint8_t> mask = build_action_mask(state, agent_id);
    if (mask.size() != stride) {
      throw std::invalid_argument("ActionParser::build_action_mask returned inconsistent sizes across agents.");
    }
    std::copy(
        mask.begin(),
        mask.end(),
        out.begin() + static_cast<std::ptrdiff_t>(agent_id * stride));
  }
}

void DoneCondition::is_done_into(
    const EnvState& state,
    int episode_ticks,
    std::span<std::uint8_t> terminated,
    std::span<std::uint8_t> truncated) const {
  if (terminated.size() != state.cars.size() || truncated.size() != state.cars.size()) {
    throw std::invalid_argument("DoneCondition::is_done_into output spans have incorrect size.");
  }

  auto [terminated_values, truncated_values] = is_done(state, episode_ticks);
  std::copy(terminated_values.begin(), terminated_values.end(), terminated.begin());
  std::copy(truncated_values.begin(), truncated_values.end(), truncated.begin());
}

}  // namespace pulsar
