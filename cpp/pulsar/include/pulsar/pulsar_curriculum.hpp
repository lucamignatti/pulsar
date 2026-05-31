// Pulsar — Phase 5: reachability-grid curriculum.
//
// Faithful port of sprint5's SelfPlayReachabilityGrid (sprint5:236-448).
//
// Design:
//   - 12×16 grid of (ball_x, ball_y) cells over [-3000,3000] × [-4000,4000].
//   - Cell status: 0=unknown, 1=frontier, 2=mastered.
//   - Sampling: frontier cells, priority = kl_running_avg + 0.1 (or 5.0 if under min_attempts).
//   - Mastery: EWMA KLD < kl_threshold after >= min_attempts.
//   - Frontier expansion: BFS 8-neighbors on mastery.
//   - Velocity tiers 0–3: stationary → full 3D.
//   - Tier promotion: all frontier cells mastered → increment tier, reset grid.
//
// KLD signal (THE DIFFERENCE from the legacy pulsar version):
//   After each PPO update, compute per-transition policy KLD as
//   `-(new_logprob - old_logprob)` (sprint5:771-793), group by cell,
//   and call record_kl(ci, cj, mean_kl_for_cell). This is NOT EGGROLL's KL.
//
// Thread safety: NOT thread-safe. The training loop owns the grid exclusively.
#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "pulsar/pulsar_env.hpp"  // PulsarEnvSnapshot, Vec3

namespace pulsar {

// ============================================================================
// PulsarReachabilityGrid
// ============================================================================

class PulsarReachabilityGrid {
 public:
  static constexpr int kXBins = 12;
  static constexpr int kYBins = 16;
  static constexpr float kXMin = -3000.0f, kXMax = 3000.0f;
  static constexpr float kYMin = -4000.0f, kYMax = 4000.0f;

  enum class CellStatus : int8_t { Unknown = 0, Frontier = 1, Mastered = 2 };

  explicit PulsarReachabilityGrid(
      float kl_threshold  = 0.015f,
      int   min_attempts  = 10,
      float ewma_alpha    = 0.1f)
      : kl_threshold_(kl_threshold),
        min_attempts_(min_attempts),
        alpha_(ewma_alpha) {
    reset_grid();
    // Seed = center cell (sprint5:265-266)
    status_[kXBins / 2][kYBins / 2] = CellStatus::Frontier;
  }

  // ---- Sampling ------------------------------------------------------------

  // Sample a start state from the current frontier (sprint5:278-394).
  // Returns (snapshot, cell_ci, cell_cj).
  // `rng` is a seeded std::mt19937.
  std::tuple<PulsarEnvSnapshot, int, int> sample_start_state(std::mt19937& rng);

  // ---- Mastery tracking ---------------------------------------------------

  // Exponential running average of per-cell KLD (sprint5:396-403).
  void record_kl(int ci, int cj, float kl_val);

  // Expand frontiers: mastered cells open 8-neighbors; tier promotion if all
  // frontiers are mastered (sprint5:405-442).
  // Returns {new_mastered, new_frontier}.
  std::pair<int, int> update_frontiers();

  // ---- State inspection ---------------------------------------------------
  int current_tier() const { return current_tier_; }
  std::array<int, 3> stats() const;  // {unknown, frontier, mastered}

  // Returns true when the entire field has been mastered at the highest tier.
  // This is the trigger to switch to standard kickoff resets.
  bool is_complete() const {
    if (current_tier_ < 3) return false;
    for (int i = 0; i < kXBins; ++i)
      for (int j = 0; j < kYBins; ++j)
        if (status_[i][j] != CellStatus::Mastered) return false;
    return true;
  }
  const int8_t (&status_raw() const)[kXBins][kYBins] {
    return reinterpret_cast<const int8_t (&)[kXBins][kYBins]>(status_);
  }
  const int32_t (&attempts_raw() const)[kXBins][kYBins] { return attempts_; }
  const double  (&kl_avg_raw() const)[kXBins][kYBins]   { return kl_avg_;   }

 private:
  void reset_grid();

  // Cell coordinate helpers
  std::pair<int, int> pos_to_cell(float x, float y) const;
  std::pair<float, float> cell_center_jitter(int ci, int cj, std::mt19937& rng) const;
  std::pair<float, float> cell_random_pos(int ci, int cj, std::mt19937& rng) const;

  float kl_threshold_;
  int   min_attempts_;
  float alpha_;
  int   current_tier_ = 0;
  int   seed_x_ = kXBins / 2;
  int   seed_y_ = kYBins / 2;

  CellStatus status_[kXBins][kYBins]{};
  int32_t    attempts_[kXBins][kYBins]{};
  double     kl_avg_[kXBins][kYBins]{};

  static constexpr float kCellW = (kXMax - kXMin) / kXBins;
  static constexpr float kCellH = (kYMax - kYMin) / kYBins;
};

// ============================================================================
// Inline implementation
// ============================================================================

inline void PulsarReachabilityGrid::reset_grid() {
  for (int i = 0; i < kXBins; ++i)
    for (int j = 0; j < kYBins; ++j) {
      status_[i][j]   = CellStatus::Unknown;
      attempts_[i][j] = 0;
      kl_avg_[i][j]   = 0.0;
    }
}

inline std::pair<int, int> PulsarReachabilityGrid::pos_to_cell(float x, float y) const {
  int i = static_cast<int>((x - kXMin) / kCellW);
  int j = static_cast<int>((y - kYMin) / kCellH);
  i = std::max(0, std::min(kXBins - 1, i));
  j = std::max(0, std::min(kYBins - 1, j));
  return {i, j};
}

inline std::pair<float, float> PulsarReachabilityGrid::cell_random_pos(
    int ci, int cj, std::mt19937& rng) const {
  std::uniform_real_distribution<float> u(0.1f, 0.9f);
  const float x = kXMin + (ci + u(rng)) * kCellW;
  const float y = kYMin + (cj + u(rng)) * kCellH;
  return {x, y};
}

inline void PulsarReachabilityGrid::record_kl(int ci, int cj, float kl_val) {
  // sprint5:396-403
  attempts_[ci][cj] += 1;
  if (attempts_[ci][cj] == 1) {
    kl_avg_[ci][cj] = static_cast<double>(kl_val);
  } else {
    kl_avg_[ci][cj] = (1.0 - alpha_) * kl_avg_[ci][cj] + alpha_ * kl_val;
  }
}

inline std::pair<int, int> PulsarReachabilityGrid::update_frontiers() {
  // sprint5:405-442
  int new_mastered = 0, new_frontier = 0;
  for (int i = 0; i < kXBins; ++i) {
    for (int j = 0; j < kYBins; ++j) {
      if (status_[i][j] != CellStatus::Frontier) continue;
      if (attempts_[i][j] >= min_attempts_ &&
          kl_avg_[i][j] <= static_cast<double>(kl_threshold_)) {
        status_[i][j] = CellStatus::Mastered;
        ++new_mastered;
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            if (di == 0 && dj == 0) continue;
            const int ni = i + di, nj = j + dj;
            if (ni >= 0 && ni < kXBins && nj >= 0 && nj < kYBins &&
                status_[ni][nj] == CellStatus::Unknown) {
              status_[ni][nj] = CellStatus::Frontier;
              ++new_frontier;
            }
          }
        }
      }
    }
  }
  // Tier promotion if no frontier cells remain
  bool any_frontier = false;
  for (int i = 0; i < kXBins && !any_frontier; ++i)
    for (int j = 0; j < kYBins && !any_frontier; ++j)
      if (status_[i][j] == CellStatus::Frontier) any_frontier = true;

  if (!any_frontier && current_tier_ < 3) {
    ++current_tier_;
    reset_grid();
    status_[seed_x_][seed_y_] = CellStatus::Frontier;
    ++new_frontier;
  }
  return {new_mastered, new_frontier};
}

inline std::array<int, 3> PulsarReachabilityGrid::stats() const {
  int unk = 0, front = 0, mast = 0;
  for (int i = 0; i < kXBins; ++i)
    for (int j = 0; j < kYBins; ++j) {
      switch (status_[i][j]) {
        case CellStatus::Unknown:  ++unk;   break;
        case CellStatus::Frontier: ++front; break;
        case CellStatus::Mastered: ++mast;  break;
      }
    }
  return {unk, front, mast};
}

inline std::tuple<PulsarEnvSnapshot, int, int>
PulsarReachabilityGrid::sample_start_state(std::mt19937& rng) {
  // Collect frontier cells (fall back to mastered, then seed)  sprint5:285-290
  std::vector<std::pair<int,int>> frontier;
  for (int i = 0; i < kXBins; ++i)
    for (int j = 0; j < kYBins; ++j)
      if (status_[i][j] == CellStatus::Frontier) frontier.push_back({i, j});
  if (frontier.empty()) {
    for (int i = 0; i < kXBins; ++i)
      for (int j = 0; j < kYBins; ++j)
        if (status_[i][j] == CellStatus::Mastered) frontier.push_back({i, j});
  }
  if (frontier.empty()) frontier.push_back({seed_x_, seed_y_});

  // Priority weights  sprint5:292-302
  std::vector<float> prio(frontier.size());
  for (std::size_t k = 0; k < frontier.size(); ++k) {
    const auto [i, j] = frontier[k];
    if (attempts_[i][j] < min_attempts_) {
      prio[k] = 5.0f;
    } else {
      prio[k] = static_cast<float>(kl_avg_[i][j]) + 0.1f;
    }
  }
  // Weighted sample
  std::discrete_distribution<int> dist(prio.begin(), prio.end());
  const int chosen = dist(rng);
  const auto [ci, cj] = frontier[chosen];

  // Ball position  sprint5:306-310
  auto [ball_x, ball_y] = cell_random_pos(ci, cj, rng);
  ball_x = std::max(-2800.0f, std::min(2800.0f, ball_x));
  ball_y = std::max(-3800.0f, std::min(3800.0f, ball_y));

  std::uniform_real_distribution<float> u_pos(-200.0f, 200.0f);
  std::uniform_real_distribution<float> u_def_x(-1500.0f, 1500.0f);
  std::uniform_real_distribution<float> u_def_y(4200.0f, 5000.0f);
  std::uniform_real_distribution<float> u_off(600.0f, 1000.0f);

  // Scorer (Blue) behind ball  sprint5:312-316
  float sc_x = std::max(-2800.0f, std::min(2800.0f, ball_x + u_pos(rng)));
  float sc_y = std::max(-4800.0f, std::min(4800.0f, ball_y - u_off(rng)));

  // Defender (Orange) in defensive half  sprint5:318-319
  const float df_x = u_def_x(rng);
  const float df_y = u_def_y(rng);

  // Velocity caps per tier  sprint5:321-365
  float b_vx = 0, b_vy = 0, b_vz = 0;
  float sc_vx = 0, sc_vy = 400, sc_vz = 0;
  float df_vx = 0, df_vy = 0, df_vz = 0;
  float b_boost = 33, o_boost = 33;

  auto ur = [&](float lo, float hi) -> float {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
  };

  switch (current_tier_) {
    case 0:
      break;
    case 1:
      b_vx = ur(-200, 200); b_vy = ur(-200, 200);
      sc_vy = ur(400, 800);
      b_boost = o_boost = 50;
      break;
    case 2:
      b_vx = ur(-500, 500); b_vy = ur(-500, 500);
      sc_vx = ur(-300, 300); sc_vy = ur(800, 1200);
      df_vx = ur(-200, 200); df_vy = ur(-500, 0);
      b_boost = o_boost = 75;
      break;
    default:  // tier 3
      b_vx = ur(-1000, 1000); b_vy = ur(-1000, 1000); b_vz = ur(0, 300);
      sc_vx = ur(-600, 600);  sc_vy = ur(1000, 1800); sc_vz = ur(0, 300);
      df_vx = ur(-500, 500);  df_vy = ur(-1000, 0);   df_vz = ur(0, 300);
      b_boost = o_boost = 100;
      break;
  }

  // Scorer faces ball  sprint5:366-369
  const float sf_dx = ball_x - sc_x, sf_dy = ball_y - sc_y;
  const float sf_n  = std::max(1e-5f, std::sqrt(sf_dx*sf_dx + sf_dy*sf_dy));
  const float sf_x = sf_dx / sf_n, sf_y_ = sf_dy / sf_n;

  // Defender faces ball  sprint5:371-375
  const float df_dx = ball_x - df_x, df_dy = ball_y - df_y;
  const float df_n  = std::max(1e-5f, std::sqrt(df_dx*df_dx + df_dy*df_dy));
  const float df_fx = df_dx / df_n, df_fy = df_dy / df_n;

  // Build snapshot  sprint5:377-394
  PulsarEnvSnapshot snap;
  snap.blue_pos    = {sc_x, sc_y, 17.0f};
  snap.blue_vel    = {sc_vx, sc_vy, sc_vz};
  snap.blue_fwd    = {sf_x, sf_y_, 0.0f};
  // right = up × fwd = (0,0,1)×(sf_x,sf_y,0) = (-sf_y, sf_x, 0)
  snap.blue_up     = {0.0f, 0.0f, 1.0f};
  snap.blue_ang_vel = {0.0f, 0.0f, 0.0f};
  snap.blue_boost  = b_boost;

  snap.orange_pos  = {df_x, df_y, 17.0f};
  snap.orange_vel  = {df_vx, df_vy, df_vz};
  snap.orange_fwd  = {df_fx, df_fy, 0.0f};
  snap.orange_up   = {0.0f, 0.0f, 1.0f};
  snap.orange_ang_vel = {0.0f, 0.0f, 0.0f};
  snap.orange_boost = o_boost;

  snap.ball_pos    = {ball_x, ball_y, 93.0f};
  snap.ball_vel    = {b_vx, b_vy, b_vz};
  snap.ball_ang_vel = {0.0f, 0.0f, 0.0f};

  return {snap, ci, cj};
}

// ============================================================================
// PulsarCurriculumMutator — integrates the grid with the collector's StateMutator
// ============================================================================
// Wraps PulsarReachabilityGrid as a StateMutator:
//   - apply() samples a start state from the grid's current frontier
//   - The caller should call record_kl / update_frontiers after each update
//   - Cell indices for each env are stored and accessible after apply()
class PulsarCurriculumMutator : public StateMutator {
 public:
  explicit PulsarCurriculumMutator(PulsarReachabilityGrid* grid, std::uint64_t seed = 0)
      : grid_(grid), rng_(seed) {}

  void apply(EnvState& state, std::uint64_t /*seed*/) const override {
    auto [snap, ci, cj] = grid_->sample_start_state(rng_);
    apply_snapshot(snap, state);
    last_ci_ = ci;
    last_cj_ = cj;
  }

  int last_ci() const { return last_ci_; }
  int last_cj() const { return last_cj_; }

 private:
  PulsarReachabilityGrid* grid_;
  mutable std::mt19937 rng_;
  mutable int last_ci_ = 0, last_cj_ = 0;
};

}  // namespace pulsar
