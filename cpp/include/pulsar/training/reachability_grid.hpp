#pragma once

#ifdef PULSAR_HAS_TORCH

#include <vector>
#include <mutex>
#include <utility>
#include <random>
#include <memory>

#include "pulsar/core/types.hpp"
#include "pulsar/rl/interfaces.hpp"
#include "pulsar/config/config.hpp"

namespace pulsar {

class SelfPlayReachabilityGrid {
 public:
  SelfPlayReachabilityGrid(
      int x_bins = 12, int y_bins = 16,
      float x_min = -3000.0F, float x_max = 3000.0F,
      float y_min = -4000.0F, float y_max = 4000.0F,
      float kl_threshold = 0.015F, int min_attempts = 10);

  // Thread-safe state sampling
  std::pair<int, int> sample_start_state(
      Vec3& ball_pos, Vec3& ball_vel, Vec3& ball_ang_vel,
      Vec3& scorer_pos, Vec3& scorer_vel, Vec3& scorer_fwd, Vec3& scorer_up,
      Vec3& defender_pos, Vec3& defender_vel, Vec3& defender_fwd, Vec3& defender_up,
      float& scorer_boost, float& defender_boost,
      std::uint64_t seed);

  // Thread-safe KLD recording
  void record_kl(int ci, int cj, float kl_val);

  // Retrieve active cell index for a seed
  int get_cell_for_seed(std::uint64_t seed);

  // Thread-safe frontier expansion
  std::pair<int, int> update_frontiers();

  // Get current stats
  void get_stats(int& out_unknown, int& out_frontier, int& out_mastered, int& out_tier) const;

  // Setters/Getters
  int current_tier() const;
  void set_tier(int tier);

 private:
  int x_bins_;
  int y_bins_;
  float x_min_;
  float x_max_;
  float y_min_;
  float y_max_;
  float cell_w_;
  float cell_h_;
  float kl_threshold_;
  int min_attempts_;

  int seed_x_;
  int seed_y_;

  mutable std::mutex mutex_;
  std::vector<std::vector<int>> status_; // 0=unknown, 1=frontier, 2=mastered
  std::vector<std::vector<int>> attempts_;
  std::vector<std::vector<double>> kl_running_avg_;
  std::unordered_map<std::uint64_t, std::pair<int, int>> active_cells_;
  int current_tier_ = 0;
};

class SelfPlayReachabilityMutator final : public StateMutator {
 public:
  explicit SelfPlayReachabilityMutator(EnvConfig config, std::shared_ptr<SelfPlayReachabilityGrid> grid);

  void apply(EnvState& state, std::uint64_t seed) const override;

 private:
  EnvConfig config_{};
  std::shared_ptr<SelfPlayReachabilityGrid> grid_;
};

extern std::shared_ptr<SelfPlayReachabilityGrid> g_reachability_grid;

}  // namespace pulsar

#endif
