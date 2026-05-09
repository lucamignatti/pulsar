#include "pulsar/logging/wandb_logger.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

namespace pulsar {
namespace {

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

}  // namespace

WandbLogger::WandbLogger(
    const WandbConfig& config,
    const std::string& run_dir,
    const std::string& config_path,
    const std::string& default_job_type)
    : config_(config), enabled_(config.enabled), run_dir_(run_dir) {
  if (!enabled_) {
    return;
  }

  namespace fs = std::filesystem;
  const std::string run_name =
      !config_.run_name.empty() ? config_.run_name : fs::path(run_dir).filename().string();
  const std::string job_type =
      !config_.job_type.empty() ? config_.job_type : default_job_type;
  const std::string wandb_dir = !config_.dir.empty() ? config_.dir : run_dir;

  std::ostringstream command;
  command << shell_quote(config_.python_executable)
          << " -u "
          << shell_quote(config_.script_path)
          << " --project " << shell_quote(config_.project)
          << " --run-dir " << shell_quote(run_dir)
          << " --run-name " << shell_quote(run_name)
          << " --job-type " << shell_quote(job_type)
          << " --wandb-dir " << shell_quote(wandb_dir)
          << " --mode " << shell_quote(config_.mode);
  if (!config_.entity.empty()) {
    command << " --entity " << shell_quote(config_.entity);
  }
  if (!config_.group.empty()) {
    command << " --group " << shell_quote(config_.group);
  }
  if (!config_path.empty()) {
    command << " --config-path " << shell_quote(config_path);
  }
  for (const auto& tag : config_.tags) {
    command << " --tag " << shell_quote(tag);
  }
  if (!config_.run_id.empty()) {
    command << " --id " << shell_quote(config_.run_id)
            << " --resume allow";
  }

  pipe_ = popen(command.str().c_str(), "w");
  if (pipe_ == nullptr) {
    throw std::runtime_error("Failed to start wandb logging process.");
  }
}

WandbLogger::~WandbLogger() {
  finish();
}

WandbLogger::WandbLogger(WandbLogger&& other) noexcept
    : config_(std::move(other.config_)),
      pipe_(other.pipe_),
      enabled_(other.enabled_) {
  other.pipe_ = nullptr;
  other.enabled_ = false;
}

WandbLogger& WandbLogger::operator=(WandbLogger&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  finish();
  config_ = std::move(other.config_);
  pipe_ = other.pipe_;
  enabled_ = other.enabled_;
  other.pipe_ = nullptr;
  other.enabled_ = false;
  return *this;
}

bool WandbLogger::enabled() const {
  return enabled_;
}

std::string WandbLogger::run_id() const {
  if (!run_id_.empty()) {
    return run_id_;
  }
  const std::filesystem::path run_id_path = std::filesystem::path(run_dir_) / "wandb_run_id.txt";
  if (!std::filesystem::exists(run_id_path)) {
    return "";
  }
  std::ifstream input(run_id_path);
  if (!input) {
    return "";
  }
  std::string id;
  input >> id;
  run_id_ = id;
  return run_id_;
}

void WandbLogger::log(const nlohmann::json& payload) {
  if (!enabled_ || pipe_ == nullptr) {
    return;
  }
  const std::string line = payload.dump();
  const std::size_t written = std::fwrite(line.data(), 1, line.size(), pipe_);
  if (written != line.size() || std::fwrite("\n", 1, 1, pipe_) != 1 || std::fflush(pipe_) != 0) {
    std::cerr << "wandb logging pipe write failed (errno " << errno << "); disabling further logging\n";
    enabled_ = false;
  }
}

void WandbLogger::finish() {
  if (pipe_ == nullptr) {
    return;
  }
  const int status = pclose(pipe_);
  pipe_ = nullptr;
  if (status != 0) {
    std::cerr << "wandb logging process exited with status " << status << '\n';
  }
}

}  // namespace pulsar
