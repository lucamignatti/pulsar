#pragma once

#include <cstdint>
#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/rl/interfaces.hpp"

#ifdef PULSAR_HAS_ROCKETSIM
namespace RocketSim {
class Arena;
class Car;
}  // namespace RocketSim
#endif

namespace pulsar {

class RocketSimTransitionEngine final : public TransitionEngine {
 public:
  RocketSimTransitionEngine(EnvConfig config, StateMutatorPtr reset_mutator);
  ~RocketSimTransitionEngine() override;

  void reset(std::uint64_t seed) override;
  StepResult step(std::span<const ControllerState> actions) override;
  void step_inplace(std::span<const ControllerState> actions) override;
  const EnvState& state() const override;
  std::size_t num_agents() const override;

 private:
  void apply_placeholder_dynamics(std::span<const ControllerState> actions);
#ifdef PULSAR_HAS_ROCKETSIM
  void sync_state_from_arena();
#endif

  EnvConfig config_{};
  StateMutatorPtr reset_mutator_{};
  EnvState state_{};
  int episode_ticks_ = 0;

#ifdef PULSAR_HAS_ROCKETSIM
  RocketSim::Arena* arena_ = nullptr;
  std::vector<RocketSim::Car*> cars_{};
  std::uint64_t last_reset_seed_ = 0;
#endif
};

}  // namespace pulsar
