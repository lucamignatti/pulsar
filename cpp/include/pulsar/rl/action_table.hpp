#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/rl/interfaces.hpp"

namespace pulsar {

class ControllerActionTable {
 public:
  ControllerActionTable() = default;
  explicit ControllerActionTable(ActionTableConfig config);

  static ActionTableConfig make_builtin(const std::string& builtin_name);

  [[nodiscard]] const ControllerState& at(std::size_t index) const;
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] const ActionTableConfig& config() const;
  [[nodiscard]] std::string hash() const;

 private:
  ActionTableConfig config_{};
};

class DiscreteActionParser final : public ActionParser {
 public:
  explicit DiscreteActionParser(ControllerActionTable action_table);

  std::vector<ControllerState> parse_actions(std::span<const std::int64_t> action_indices) const override;
  void parse_actions_into(
      std::span<const std::int64_t> action_indices,
      std::span<ControllerState> out) const override;
  [[nodiscard]] const ControllerActionTable& action_table() const;

 private:
  ControllerActionTable action_table_{};
};

}  // namespace pulsar
