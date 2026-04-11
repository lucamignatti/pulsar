#pragma once

#include "pulsar/config/config.hpp"
#include "pulsar/rl/interfaces.hpp"

namespace pulsar {

class CombinedRewardFunction final : public RewardFunction {
 public:
  explicit CombinedRewardFunction(RewardConfig config);

  std::vector<float> get_rewards(
      const EnvState& previous_state,
      const EnvState& current_state,
      std::span<const std::uint8_t> terminated,
      std::span<const std::uint8_t> truncated) const override;

 private:
  RewardConfig config_{};
};

}  // namespace pulsar

