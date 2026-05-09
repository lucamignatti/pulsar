#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace pulsar {

class PositionHeatmapLogger {
 public:
  PositionHeatmapLogger();
  ~PositionHeatmapLogger();

  PositionHeatmapLogger(const PositionHeatmapLogger&) = delete;
  PositionHeatmapLogger& operator=(const PositionHeatmapLogger&) = delete;
  PositionHeatmapLogger(PositionHeatmapLogger&&) = delete;
  PositionHeatmapLogger& operator=(PositionHeatmapLogger&&) = delete;

  void start();
  void stop();

  bool enabled() const;

  void record_positions(
      const float* pos_x,
      const float* pos_y,
      const float* teams,
      int count);

  void record_positions_interleaved(const float* car_data, int count);

  void on_update_complete();

  bool try_get_payload(nlohmann::json& out);

 private:
  void worker_loop();

  static constexpr int kCols = 24;
  static constexpr int kRows = 18;
  static constexpr float kXMin = -4600.0F;
  static constexpr float kXMax = 4600.0F;
  static constexpr float kYMin = -5120.0F;
  static constexpr float kYMax = 5120.0F;
  static constexpr double kLogIntervalSeconds = 60.0;

  int grid_size_ = kCols * kRows;

  std::vector<int64_t> blue_grid_;
  std::vector<int64_t> orange_grid_;
  int64_t step_count_ = 0;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<int64_t> blue_snapshot_;
  std::vector<int64_t> orange_snapshot_;
  int64_t snapshot_step_count_ = 0;
  bool has_snapshot_ = false;

  std::mutex payload_mutex_;
  std::string payload_str_;
  bool payload_ready_ = false;

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::atomic<bool> enabled_{false};

  std::chrono::steady_clock::time_point last_log_time_;
};

}  // namespace pulsar
