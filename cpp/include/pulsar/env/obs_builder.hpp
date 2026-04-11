#pragma once

#include "pulsar/config/config.hpp"
#include "pulsar/rl/interfaces.hpp"

namespace pulsar {

class PulsarObsBuilder final : public ObsBuilder {
 public:
  explicit PulsarObsBuilder(EnvConfig config);

  std::vector<float> build_obs(const EnvState& state, AgentId agent_id) const override;
  void build_obs_batch(const EnvState& state, std::span<float> out) const override;
  std::size_t obs_dim() const override;

 private:
  EnvConfig config_{};
};

}  // namespace pulsar
