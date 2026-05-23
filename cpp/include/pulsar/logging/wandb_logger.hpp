#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pulsar/config/config.hpp"

namespace pulsar {

class WandbLogger {
 public:
  WandbLogger(
      const WandbConfig& config,
      const std::string& run_dir,
      const std::string& config_path,
      const std::string& default_job_type);
  ~WandbLogger();

  WandbLogger(const WandbLogger&) = delete;
  WandbLogger& operator=(const WandbLogger&) = delete;
  WandbLogger(WandbLogger&& other) noexcept;
  WandbLogger& operator=(WandbLogger&& other) noexcept;

  [[nodiscard]] bool enabled() const;
  [[nodiscard]] std::string run_id() const;

  // Decoupled, first-class logger interface
  void add_metric(const std::string& section, const std::string& key, nlohmann::json value);
  void add_table(
      const std::string& key,
      const std::vector<std::string>& columns,
      const std::vector<std::vector<nlohmann::json>>& data);
  void commit(std::int64_t global_step);

  void log(const nlohmann::json& payload);
  void finish();

 private:
  WandbConfig config_{};
  std::FILE* pipe_ = nullptr;
  bool enabled_ = false;
  std::string run_dir_{};
  mutable std::string run_id_{};

  // Logging buffers
  nlohmann::json payload_ = nlohmann::json::object();
  nlohmann::json sections_ = nlohmann::json::object();
};

}  // namespace pulsar
