#include "pulsar/logging/position_heatmap_logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <nlohmann/json.hpp>

namespace pulsar {

PositionHeatmapLogger::PositionHeatmapLogger()
    : blue_grid_(static_cast<std::size_t>(grid_size_), 0),
      orange_grid_(static_cast<std::size_t>(grid_size_), 0),
      blue_snapshot_(static_cast<std::size_t>(grid_size_), 0),
      orange_snapshot_(static_cast<std::size_t>(grid_size_), 0),
      last_log_time_(std::chrono::steady_clock::now()) {}

PositionHeatmapLogger::~PositionHeatmapLogger() {
  stop();
}

void PositionHeatmapLogger::start() {
  if (running_.exchange(true)) {
    return;
  }
  enabled_.store(true);
  worker_ = std::thread(&PositionHeatmapLogger::worker_loop, this);
}

void PositionHeatmapLogger::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  enabled_.store(false);
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

bool PositionHeatmapLogger::enabled() const {
  return enabled_.load();
}

void PositionHeatmapLogger::record_positions(
    const float* pos_x,
    const float* pos_y,
    const float* teams,
    int count) {
  if (count <= 0) {
    return;
  }

  const float col_scale = static_cast<float>(kCols) / (kXMax - kXMin);
  const float row_scale = static_cast<float>(kRows) / (kYMax - kYMin);

  for (int i = 0; i < count; ++i) {
    const float x = pos_x[i];
    const float y = pos_y[i];
    const int team = static_cast<int>(teams[i]);

    const int col = std::clamp(
        static_cast<int>((x - kXMin) * col_scale), 0, kCols - 1);
    const int row = std::clamp(
        static_cast<int>((y - kYMin) * row_scale), 0, kRows - 1);
    const int idx = row * kCols + col;

    if (team == 0) {
      blue_grid_[static_cast<std::size_t>(idx)]++;
    } else {
      orange_grid_[static_cast<std::size_t>(idx)]++;
    }
  }
  step_count_ += static_cast<int64_t>(count);
}

void PositionHeatmapLogger::record_positions_interleaved(const float* car_data, int count) {
  if (count <= 0) {
    return;
  }

  const float col_scale = static_cast<float>(kCols) / (kXMax - kXMin);
  const float row_scale = static_cast<float>(kRows) / (kYMax - kYMin);

  for (int i = 0; i < count; ++i) {
    const float x = car_data[i * 4 + 0];
    const float y = car_data[i * 4 + 1];
    const int team = static_cast<int>(car_data[i * 4 + 3]);

    const int col = std::clamp(
        static_cast<int>((x - kXMin) * col_scale), 0, kCols - 1);
    const int row = std::clamp(
        static_cast<int>((y - kYMin) * row_scale), 0, kRows - 1);
    const int idx = row * kCols + col;

    if (team == 0) {
      blue_grid_[static_cast<std::size_t>(idx)]++;
    } else {
      orange_grid_[static_cast<std::size_t>(idx)]++;
    }
  }
  step_count_ += static_cast<int64_t>(count);
}

void PositionHeatmapLogger::on_update_complete() {
  if (!enabled_.load()) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::copy(blue_grid_.begin(), blue_grid_.end(), blue_snapshot_.begin());
    std::copy(orange_grid_.begin(), orange_grid_.end(), orange_snapshot_.begin());
    std::fill(blue_grid_.begin(), blue_grid_.end(), 0);
    std::fill(orange_grid_.begin(), orange_grid_.end(), 0);
    snapshot_step_count_ = step_count_;
    step_count_ = 0;
    has_snapshot_ = true;
  }
  cv_.notify_one();
}

bool PositionHeatmapLogger::try_get_payload(nlohmann::json& out) {
  std::lock_guard<std::mutex> lock(payload_mutex_);
  if (!payload_ready_) {
    return false;
  }
  out = nlohmann::json::parse(payload_str_);
  payload_str_.clear();
  payload_ready_ = false;
  return true;
}

void PositionHeatmapLogger::worker_loop() {
  while (running_.load()) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::seconds(1), [this] {
        return !running_.load() || has_snapshot_;
      });

      if (!running_.load()) {
        return;
      }

      const auto now = std::chrono::steady_clock::now();
      const double elapsed =
          std::chrono::duration<double>(now - last_log_time_).count();
      if (elapsed < kLogIntervalSeconds && has_snapshot_) {
        continue;
      }

      if (!has_snapshot_ || snapshot_step_count_ <= 0) {
        continue;
      }

      last_log_time_ = now;

      std::vector<int64_t> blue_snap = blue_snapshot_;
      std::vector<int64_t> orange_snap = orange_snapshot_;
      int64_t snap_steps = snapshot_step_count_;

      has_snapshot_ = false;
      lock.unlock();

      const float col_width = (kXMax - kXMin) / static_cast<float>(kCols);
      const float row_width = (kYMax - kYMin) / static_cast<float>(kRows);

      nlohmann::json x_labels = nlohmann::json::array();
      for (int c = 0; c < kCols; ++c) {
        x_labels.push_back(kXMin + (static_cast<float>(c) + 0.5F) * col_width);
      }

      nlohmann::json y_labels = nlohmann::json::array();
      for (int r = 0; r < kRows; ++r) {
        y_labels.push_back(kYMin + (static_cast<float>(r) + 0.5F) * row_width);
      }

      nlohmann::json blue_rows = nlohmann::json::array();
      nlohmann::json orange_rows = nlohmann::json::array();
      for (int r = 0; r < kRows; ++r) {
        nlohmann::json blue_row = nlohmann::json::array();
        nlohmann::json orange_row = nlohmann::json::array();
        for (int c = 0; c < kCols; ++c) {
          blue_row.push_back(blue_snap[static_cast<std::size_t>(r * kCols + c)]);
          orange_row.push_back(orange_snap[static_cast<std::size_t>(r * kCols + c)]);
        }
        blue_rows.push_back(blue_row);
        orange_rows.push_back(orange_row);
      }

      nlohmann::json heatmap;
      heatmap["x_labels"] = x_labels;
      heatmap["y_labels"] = y_labels;
      heatmap["blue_grid"] = blue_rows;
      heatmap["orange_grid"] = orange_rows;
      heatmap["step_count"] = snap_steps;

      {
        std::lock_guard<std::mutex> plock(payload_mutex_);
        nlohmann::json wrapper;
        wrapper["_position_heatmap"] = heatmap;
        payload_str_ = wrapper.dump();
        payload_ready_ = true;
      }
    }
  }
}

}  // namespace pulsar
