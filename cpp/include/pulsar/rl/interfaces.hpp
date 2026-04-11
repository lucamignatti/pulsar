#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pulsar/core/types.hpp"

namespace pulsar {

using AgentId = std::size_t;

class TransitionEngine {
 public:
  virtual ~TransitionEngine() = default;

  virtual void reset(std::uint64_t seed) = 0;
  virtual StepResult step(std::span<const ControllerState> actions) = 0;
  virtual const EnvState& state() const = 0;
  virtual std::size_t num_agents() const = 0;
};

class StateMutator {
 public:
  virtual ~StateMutator() = default;
  virtual void apply(EnvState& state, std::uint64_t seed) const = 0;
};

class ObsBuilder {
 public:
  virtual ~ObsBuilder() = default;
  virtual std::vector<float> build_obs(const EnvState& state, AgentId agent_id) const = 0;
  virtual std::size_t obs_dim() const = 0;
};

class ActionParser {
 public:
  virtual ~ActionParser() = default;
  virtual std::vector<ControllerState> parse_actions(std::span<const std::int64_t> action_indices) const = 0;
};

class RewardFunction {
 public:
  virtual ~RewardFunction() = default;
  virtual std::vector<float> get_rewards(
      const EnvState& previous_state,
      const EnvState& current_state,
      std::span<const std::uint8_t> terminated,
      std::span<const std::uint8_t> truncated) const = 0;
};

class DoneCondition {
 public:
  virtual ~DoneCondition() = default;
  virtual std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>> is_done(
      const EnvState& state,
      int episode_ticks) const = 0;
};

using TransitionEnginePtr = std::shared_ptr<TransitionEngine>;
using StateMutatorPtr = std::shared_ptr<StateMutator>;
using ObsBuilderPtr = std::shared_ptr<ObsBuilder>;
using ActionParserPtr = std::shared_ptr<ActionParser>;
using RewardFunctionPtr = std::shared_ptr<RewardFunction>;
using DoneConditionPtr = std::shared_ptr<DoneCondition>;

}  // namespace pulsar

