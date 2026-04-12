#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include "pulsar/config/config.hpp"

namespace pulsar::test {

inline void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

inline std::filesystem::path find_repo_path(const std::string& leaf_name) {
  namespace fs = std::filesystem;
  fs::path current = fs::current_path();
  for (int depth = 0; depth < 8; ++depth) {
    const fs::path candidate = current / leaf_name;
    if (fs::exists(candidate)) {
      return fs::canonical(candidate);
    }
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }
  throw std::runtime_error("failed to locate repo path: " + leaf_name);
}

inline std::filesystem::path find_repo_collision_meshes() {
  return find_repo_path("collision_meshes");
}

inline ExperimentConfig make_test_config() {
  ExperimentConfig config;
  config.reward.terms = {
      {"goal", 10.0F},
      {"touch", 0.25F},
      {"speed_to_ball", 0.01F},
      {"ball_to_goal", 0.1F},
      {"face_ball", 0.005F},
  };
  config.reward.team_spirit = 0.2F;
  config.reward.opponent_scale = 1.0F;
  config.action_table.builtin = "rlgym_lookup_v1";
  return config;
}

}  // namespace pulsar::test
