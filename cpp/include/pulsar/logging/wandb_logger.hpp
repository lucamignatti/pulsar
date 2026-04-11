#pragma once

#include <cstdio>
#include <string>

#include <nlohmann/json_fwd.hpp>

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
  void log(const nlohmann::json& payload);
  void finish();

 private:
  WandbConfig config_{};
  std::FILE* pipe_ = nullptr;
  bool enabled_ = false;
};

}  // namespace pulsar
