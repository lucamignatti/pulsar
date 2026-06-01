#pragma once

#include <cstdint>
#include <string>

#include "pulsar/core/types.hpp"

namespace pulsar {

struct EnvConfig {
  std::string mode = "soccar";
  std::string collision_meshes_path = "collision_meshes";
  int team_size = 2;
  int tick_skip = 8;
  bool randomize_kickoffs = true;
  std::uint64_t seed = 0;
};

}  // namespace pulsar
