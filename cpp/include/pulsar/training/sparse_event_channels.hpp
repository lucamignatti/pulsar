#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace pulsar {

struct SparseEventChannelSpec {
  std::string_view name;
  int default_horizon;
  float default_weight;
};

inline constexpr std::array<SparseEventChannelSpec, 20> kSparseEventChannels{{
    {"goal", 40, 0.05F},
    {"ball_touch", 15, 0.01F},
    {"air_touch", 20, 0.02F},
    {"flip_reset", 40, 0.05F},
    {"boost_pickup", 20, 0.01F},
    {"big_boost_pickup", 30, 0.015F},
    {"bump", 20, 0.015F},
    {"demo", 30, 0.03F},
    {"speed_flip", 25, 0.02F},
    {"wavedash", 30, 0.02F},
    {"wall_dash", 30, 0.02F},
    {"flick", 30, 0.03F},
    {"air_dribble", 35, 0.03F},
    {"ceiling_shot", 45, 0.04F},
    {"double_tap", 45, 0.04F},
    {"preflip", 30, 0.025F},
    {"redirect", 35, 0.03F},
    {"pogo", 35, 0.025F},
    {"pinch", 35, 0.03F},
    {"kickoff_50_50", 20, 0.015F},
}};

inline constexpr std::size_t kSparseEventChannelCount = kSparseEventChannels.size();

inline int sparse_event_channel_index(std::string_view name) {
  for (std::size_t i = 0; i < kSparseEventChannels.size(); ++i) {
    if (kSparseEventChannels[i].name == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace pulsar
