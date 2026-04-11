#include "pulsar/rl/interfaces.hpp"

#include <stdexcept>

namespace pulsar {

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

}  // namespace pulsar
