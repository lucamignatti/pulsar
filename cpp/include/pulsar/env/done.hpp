#pragma once

#include "pulsar/config/config.hpp"
#include "pulsar/rl/interfaces.hpp"

namespace pulsar {

class SimpleDoneCondition final : public DoneCondition {
 public:
  explicit SimpleDoneCondition(EnvConfig config);

  std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>> is_done(
      const EnvState& state,
      int episode_ticks) const override;

 private:
  EnvConfig config_{};
};

}  // namespace pulsar

