#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace pulsar {

struct SparseEventChannelSpec {
  std::string_view name;
  int default_horizon;
};

inline constexpr std::array<SparseEventChannelSpec, 20> kSparseEventChannels{{
    {"goal", 40},
    {"ball_touch", 15},
    {"air_touch", 20},
    {"flip_reset", 40},
    {"boost_pickup", 20},
    {"big_boost_pickup", 30},
    {"bump", 20},
    {"demo", 30},
    {"speed_flip", 25},
    {"wavedash", 30},
    {"wall_dash", 30},
    {"flick", 30},
    {"air_dribble", 35},
    {"ceiling_shot", 45},
    {"double_tap", 45},
    {"preflip", 30},
    {"redirect", 35},
    {"pogo", 35},
    {"pinch", 35},
    {"kickoff_50_50", 20},
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
