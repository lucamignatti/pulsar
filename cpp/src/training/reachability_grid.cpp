#include "pulsar/training/reachability_grid.hpp"

#ifdef PULSAR_HAS_TORCH

#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace pulsar {

SelfPlayReachabilityGrid::SelfPlayReachabilityGrid(
    int x_bins, int y_bins,
    float x_min, float x_max,
    float y_min, float y_max,
    float kl_threshold, int min_attempts)
    : x_bins_(x_bins),
      y_bins_(y_bins),
      x_min_(x_min),
      x_max_(x_max),
      y_min_(y_min),
      y_max_(y_max),
      kl_threshold_(kl_threshold),
      min_attempts_(min_attempts) {
  cell_w_ = (x_max_ - x_min_) / static_cast<float>(x_bins_);
  cell_h_ = (y_max_ - y_min_) / static_cast<float>(y_bins_);

  seed_x_ = x_bins_ / 2;
  seed_y_ = y_bins_ / 2;

  status_.assign(x_bins_, std::vector<int>(y_bins_, 0));
  attempts_.assign(x_bins_, std::vector<int>(y_bins_, 0));
  kl_running_avg_.assign(x_bins_, std::vector<double>(y_bins_, 0.0));

  status_[seed_x_][seed_y_] = 1; // seed cell is frontier
}

std::pair<int, int> SelfPlayReachabilityGrid::sample_start_state(
    Vec3& ball_pos, Vec3& ball_vel, Vec3& ball_ang_vel,
    Vec3& scorer_pos, Vec3& scorer_vel, Vec3& scorer_fwd, Vec3& scorer_up,
    Vec3& defender_pos, Vec3& defender_vel, Vec3& defender_fwd, Vec3& defender_up,
    float& scorer_boost, float& defender_boost,
    std::uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::mt19937_64 rng(seed);
  auto rand_range = [&](float min_val, float max_val) -> float {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    return dist(rng);
  };

  // 1. Gather candidate cells (first try frontier, then mastered, then seed)
  std::vector<std::pair<int, int>> candidates;
  for (int i = 0; i < x_bins_; ++i) {
    for (int j = 0; j < y_bins_; ++j) {
      if (status_[i][j] == 1) {
        candidates.emplace_back(i, j);
      }
    }
  }

  if (candidates.empty()) {
    for (int i = 0; i < x_bins_; ++i) {
      for (int j = 0; j < y_bins_; ++j) {
        if (status_[i][j] == 2) {
          candidates.emplace_back(i, j);
        }
      }
    }
  }

  if (candidates.empty()) {
    candidates.emplace_back(seed_x_, seed_y_);
  }

  // 2. Compute priorities
  std::vector<float> priorities;
  priorities.reserve(candidates.size());
  float sum_priorities = 0.0F;

  for (const auto& cell : candidates) {
    int i = cell.first;
    int j = cell.second;
    int att = attempts_[i][j];
    float p = 0.0F;
    if (att < min_attempts_) {
      p = 5.0F;
    } else {
      p = static_cast<float>(kl_running_avg_[i][j]) + 0.1F;
    }
    priorities.push_back(p);
    sum_priorities += p;
  }

  // 3. Sample a cell
  std::pair<int, int> chosen_cell = candidates.front();
  if (sum_priorities > 0.0F) {
    std::uniform_real_distribution<float> dist(0.0F, sum_priorities);
    float target = dist(rng);
    float accum = 0.0F;
    for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
      accum += priorities[idx];
      if (accum >= target) {
        chosen_cell = candidates[idx];
        break;
      }
    }
  }

  int ci = chosen_cell.first;
  int cj = chosen_cell.second;

  active_cells_[seed] = chosen_cell;

  // 4. Sample ball position within chosen cell
  float cell_x_min = x_min_ + static_cast<float>(ci) * cell_w_;
  float cell_y_min = y_min_ + static_cast<float>(cj) * cell_h_;

  ball_pos.x = std::clamp(cell_x_min + rand_range(0.1F, 0.9F) * cell_w_, -2800.0F, 2800.0F);
  ball_pos.y = std::clamp(cell_y_min + rand_range(0.1F, 0.9F) * cell_h_, -3800.0F, 3800.0F);
  ball_pos.z = 93.0F;

  // 5. Scorer (Blue) behind the ball
  scorer_pos.y = std::clamp(ball_pos.y - rand_range(600.0F, 1000.0F), -4800.0F, 4800.0F);
  scorer_pos.x = std::clamp(ball_pos.x + rand_range(-200.0F, 200.0F), -2800.0F, 2800.0F);
  scorer_pos.z = 17.0F;

  // 6. Defender (Orange) randomized in own half (forces goalie movement & recovery saves)
  defender_pos.y = rand_range(4200.0F, 5000.0F);
  defender_pos.x = rand_range(-1500.0F, 1500.0F);
  defender_pos.z = 17.0F;

  // 7. Setup velocities based on current tier
  ball_ang_vel = {0.0F, 0.0F, 0.0F};
  scorer_vel = {0.0F, 0.0F, 0.0F};
  defender_vel = {0.0F, 0.0F, 0.0F};

  if (current_tier_ == 0) {
    ball_vel = {0.0F, 0.0F, 0.0F};
    scorer_vel = {0.0F, 400.0F, 0.0F};
    defender_vel = {0.0F, 0.0F, 0.0F};
    scorer_boost = 33.0F;
    defender_boost = 33.0F;
  } else if (current_tier_ == 1) {
    ball_vel = {rand_range(-200.0F, 200.0F), rand_range(-200.0F, 200.0F), 0.0F};
    scorer_vel = {0.0F, rand_range(400.0F, 800.0F), 0.0F};
    defender_vel = {0.0F, 0.0F, 0.0F};
    scorer_boost = 50.0F;
    defender_boost = 50.0F;
  } else if (current_tier_ == 2) {
    ball_vel = {rand_range(-500.0F, 500.0F), rand_range(-500.0F, 500.0F), 0.0F};
    scorer_vel = {rand_range(-300.0F, 300.0F), rand_range(800.0F, 1200.0F), 0.0F};
    defender_vel = {rand_range(-200.0F, 200.0F), rand_range(-500.0F, 0.0F), 0.0F}; // coming out of net
    scorer_boost = 75.0F;
    defender_boost = 75.0F;
  } else {
    ball_vel = {rand_range(-1000.0F, 1000.0F), rand_range(-1000.0F, 1000.0F), rand_range(0.0F, 300.0F)};
    scorer_vel = {rand_range(-600.0F, 600.0F), rand_range(1000.0F, 1800.0F), rand_range(0.0F, 300.0F)};
    defender_vel = {rand_range(-500.0F, 500.0F), rand_range(-1000.0F, 0.0F), rand_range(0.0F, 300.0F)};
    scorer_boost = 100.0F;
    defender_boost = 100.0F;
  }

  // 8. Align orientation to face the ball
  float dx = ball_pos.x - scorer_pos.x;
  float dy = ball_pos.y - scorer_pos.y;
  float norm = std::max(1.0e-5F, std::sqrt(dx * dx + dy * dy));
  scorer_fwd = {dx / norm, dy / norm, 0.0F};
  scorer_up = {0.0F, 0.0F, 1.0F};

  float ddx = ball_pos.x - defender_pos.x;
  float ddy = ball_pos.y - defender_pos.y;
  float dnorm = std::max(1.0e-5F, std::sqrt(ddx * ddx + ddy * ddy));
  defender_fwd = {ddx / dnorm, ddy / dnorm, 0.0F};
  defender_up = {0.0F, 0.0F, 1.0F};

  return chosen_cell;
}

int SelfPlayReachabilityGrid::get_cell_for_seed(std::uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = active_cells_.find(seed);
  if (it != active_cells_.end()) {
    return it->second.first * y_bins_ + it->second.second;
  }
  return seed_x_ * y_bins_ + seed_y_;
}

void SelfPlayReachabilityGrid::record_kl(int ci, int cj, float kl_val) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ci < 0 || ci >= x_bins_ || cj < 0 || cj >= y_bins_) return;

  attempts_[ci][cj]++;
  if (attempts_[ci][cj] == 1) {
    kl_running_avg_[ci][cj] = kl_val;
  } else {
    kl_running_avg_[ci][cj] = 0.9 * kl_running_avg_[ci][cj] + 0.1 * kl_val;
  }
}

std::pair<int, int> SelfPlayReachabilityGrid::update_frontiers() {
  std::lock_guard<std::mutex> lock(mutex_);

  int new_mastered = 0;
  int new_frontier = 0;

  for (int i = 0; i < x_bins_; ++i) {
    for (int j = 0; j < y_bins_; ++j) {
      if (status_[i][j] != 1) continue; // Only process frontier cells

      int att = attempts_[i][j];
      if (att >= min_attempts_) {
        double kl = kl_running_avg_[i][j];
        if (kl <= static_cast<double>(kl_threshold_)) {
          status_[i][j] = 2; // Mastered!
          new_mastered++;

          // Expand to adjacent cells
          for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
              if (di == 0 && dj == 0) continue;
              int ni = i + di;
              int nj = j + dj;
              if (ni >= 0 && ni < x_bins_ && nj >= 0 && nj < y_bins_) {
                if (status_[ni][nj] == 0) {
                  status_[ni][nj] = 1;
                  new_frontier++;
                }
              }
            }
          }
        }
      }
    }
  }

  // Check for tier promotion
  bool has_frontier = false;
  for (int i = 0; i < x_bins_; ++i) {
    for (int j = 0; j < y_bins_; ++j) {
      if (status_[i][j] == 1) {
        has_frontier = true;
        break;
      }
    }
    if (has_frontier) break;
  }

  if (!has_frontier && current_tier_ < 3) {
    current_tier_++;
    std::cout << "\n[TIER PROMOTION] Promoting C++ Self-Play Grid to velocity tier " << current_tier_ << "!\n" << std::flush;
    
    // Reset grid status but keep seed as active frontier
    for (int i = 0; i < x_bins_; ++i) {
      for (int j = 0; j < y_bins_; ++j) {
        status_[i][j] = 0;
        attempts_[i][j] = 0;
        kl_running_avg_[i][j] = 0.0;
      }
    }
    status_[seed_x_][seed_y_] = 1;
    new_frontier++;
  }

  return {new_mastered, new_frontier};
}

void SelfPlayReachabilityGrid::get_stats(int& out_unknown, int& out_frontier, int& out_mastered, int& out_tier) const {
  std::lock_guard<std::mutex> lock(mutex_);
  out_unknown = 0;
  out_frontier = 0;
  out_mastered = 0;
  out_tier = current_tier_;

  for (int i = 0; i < x_bins_; ++i) {
    for (int j = 0; j < y_bins_; ++j) {
      if (status_[i][j] == 0) out_unknown++;
      else if (status_[i][j] == 1) out_frontier++;
      else if (status_[i][j] == 2) out_mastered++;
    }
  }
}

int SelfPlayReachabilityGrid::current_tier() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_tier_;
}

void SelfPlayReachabilityGrid::set_tier(int tier) {
  std::lock_guard<std::mutex> lock(mutex_);
  current_tier_ = tier;
}


SelfPlayReachabilityMutator::SelfPlayReachabilityMutator(
    EnvConfig config,
    std::shared_ptr<SelfPlayReachabilityGrid> grid)
    : config_(std::move(config)), grid_(std::move(grid)) {}

void SelfPlayReachabilityMutator::apply(EnvState& state, std::uint64_t seed) const {
  if (state.cars.size() < 2) {
    state.cars.resize(2);
  }

  state.cars[0].id = 0;
  state.cars[0].team = Team::Blue;
  state.cars[1].id = 1;
  state.cars[1].team = Team::Orange;

  state.boost_pad_timers.assign(34, 0.0F);

  Vec3 ball_pos, ball_vel, ball_ang_vel;
  Vec3 scorer_pos, scorer_vel, scorer_fwd, scorer_up;
  Vec3 defender_pos, defender_vel, defender_fwd, defender_up;
  float scorer_boost = 33.0F;
  float defender_boost = 33.0F;

  grid_->sample_start_state(
      ball_pos, ball_vel, ball_ang_vel,
      scorer_pos, scorer_vel, scorer_fwd, scorer_up,
      defender_pos, defender_vel, defender_fwd, defender_up,
      scorer_boost, defender_boost,
      seed);

  // Sync ball state
  state.ball.position = ball_pos;
  state.ball.velocity = ball_vel;
  state.ball.angular_velocity = ball_ang_vel;

  // Sync scorer (Blue)
  state.cars[0].position = scorer_pos;
  state.cars[0].velocity = scorer_vel;
  state.cars[0].forward = scorer_fwd;
  state.cars[0].up = scorer_up;
  state.cars[0].boost = scorer_boost;
  state.cars[0].angular_velocity = {0.0F, 0.0F, 0.0F};
  state.cars[0].on_ground = true;
  state.cars[0].has_flip = true;
  state.cars[0].has_flip_reset = false;

  // Sync defender (Orange)
  state.cars[1].position = defender_pos;
  state.cars[1].velocity = defender_vel;
  state.cars[1].forward = defender_fwd;
  state.cars[1].up = defender_up;
  state.cars[1].boost = defender_boost;
  state.cars[1].angular_velocity = {0.0F, 0.0F, 0.0F};
  state.cars[1].on_ground = true;
  state.cars[1].has_flip = true;
  state.cars[1].has_flip_reset = false;

  state.goal_scored = false;
  state.kickoff_pause = false;
  state.last_touch_agent = -1;
  state.last_touch_tick = 0;
  state.tick = 0;
}

std::shared_ptr<SelfPlayReachabilityGrid> g_reachability_grid = nullptr;

} // namespace pulsar

#endif
