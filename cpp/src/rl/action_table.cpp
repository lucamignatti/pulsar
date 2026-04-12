#include "pulsar/rl/action_table.hpp"

#include <algorithm>
#include <stdexcept>

namespace pulsar {
namespace {

std::vector<ControllerState> make_rlgym_lookup_actions() {
  std::vector<ControllerState> actions;

  for (int throttle : {-1, 0, 1}) {
    for (int steer : {-1, 0, 1}) {
      for (int boost : {0, 1}) {
        for (int handbrake : {0, 1}) {
          if (boost == 1 && throttle != 1) {
            continue;
          }
          actions.push_back(ControllerState{
              .throttle = static_cast<float>(throttle != 0 ? throttle : boost),
              .steer = static_cast<float>(steer),
              .yaw = 0.0F,
              .pitch = static_cast<float>(steer),
              .roll = 0.0F,
              .jump = false,
              .boost = boost == 1,
              .handbrake = handbrake == 1,
          });
        }
      }
    }
  }

  for (int pitch : {-1, 0, 1}) {
    for (int yaw : {-1, 0, 1}) {
      for (int roll : {-1, 0, 1}) {
        for (int jump : {0, 1}) {
          for (int boost : {0, 1}) {
            if (jump == 1 && yaw != 0) {
              continue;
            }
            if (pitch == 0 && roll == 0 && jump == 0) {
              continue;
            }

            const bool handbrake = jump == 1 && (pitch != 0 || yaw != 0 || roll != 0);
            actions.push_back(ControllerState{
                .throttle = static_cast<float>(boost),
                .steer = static_cast<float>(yaw),
                .yaw = static_cast<float>(yaw),
                .pitch = static_cast<float>(pitch),
                .roll = static_cast<float>(roll),
                .jump = jump == 1,
                .boost = boost == 1,
                .handbrake = handbrake,
            });
          }
        }
      }
    }
  }

  return actions;
}

}  // namespace

ControllerActionTable::ControllerActionTable(ActionTableConfig config) : config_(std::move(config)) {
  if (config_.actions.empty() && !config_.builtin.empty()) {
    config_ = make_builtin(config_.builtin);
  }
  if (config_.actions.empty()) {
    throw std::invalid_argument("Action table must contain at least one action.");
  }
}

ActionTableConfig ControllerActionTable::make_builtin(const std::string& builtin_name) {
  if (builtin_name == "rlgym_lookup_v1") {
    return ActionTableConfig{
        .builtin = builtin_name,
        .actions = make_rlgym_lookup_actions(),
    };
  }
  throw std::invalid_argument("Unknown builtin action table: " + builtin_name);
}

const ControllerState& ControllerActionTable::at(std::size_t index) const {
  if (index >= config_.actions.size()) {
    throw std::out_of_range("Action index out of range.");
  }
  return config_.actions[index];
}

std::size_t ControllerActionTable::size() const {
  return config_.actions.size();
}

const ActionTableConfig& ControllerActionTable::config() const {
  return config_;
}

std::string ControllerActionTable::hash() const {
  return action_table_hash(config_);
}

DiscreteActionParser::DiscreteActionParser(ControllerActionTable action_table)
    : action_table_(std::move(action_table)) {}

std::vector<ControllerState> DiscreteActionParser::parse_actions(
    std::span<const std::int64_t> action_indices) const {
  std::vector<ControllerState> actions;
  actions.reserve(action_indices.size());

  for (const std::int64_t index : action_indices) {
    const std::size_t bounded_index = static_cast<std::size_t>(
        std::clamp<std::int64_t>(index, 0, static_cast<std::int64_t>(action_table_.size() - 1)));
    actions.push_back(action_table_.at(bounded_index));
  }

  return actions;
}

void DiscreteActionParser::parse_actions_into(
    std::span<const std::int64_t> action_indices,
    std::span<ControllerState> out) const {
  if (out.size() != action_indices.size()) {
    throw std::invalid_argument("DiscreteActionParser::parse_actions_into output span has incorrect size.");
  }

  for (std::size_t i = 0; i < action_indices.size(); ++i) {
    const std::size_t bounded_index = static_cast<std::size_t>(
        std::clamp<std::int64_t>(action_indices[i], 0, static_cast<std::int64_t>(action_table_.size() - 1)));
    out[i] = action_table_.at(bounded_index);
  }
}

std::vector<std::uint8_t> DiscreteActionParser::build_action_mask(const EnvState& state, AgentId agent_id) const {
  if (agent_id >= state.cars.size()) {
    throw std::out_of_range("Action mask agent index out of range.");
  }

  const CarState& car = state.cars[agent_id];
  std::vector<std::uint8_t> mask(action_table_.size(), static_cast<std::uint8_t>(1));
  for (std::size_t index = 0; index < action_table_.size(); ++index) {
    const ControllerState& action = action_table_.at(index);
    bool valid = true;

    if (action.boost && car.boost <= 0.5F) {
      valid = false;
    }
    if (action.jump && !(car.on_ground || car.has_flip)) {
      valid = false;
    }

    mask[index] = static_cast<std::uint8_t>(valid ? 1 : 0);
  }
  return mask;
}

const ControllerActionTable& DiscreteActionParser::action_table() const {
  return action_table_;
}

}  // namespace pulsar
