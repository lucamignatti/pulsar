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

// Drop log entries if the async queue grows beyond this to prevent unbounded
// memory growth when the wandb subprocess stalls.
constexpr std::size_t kMaxQueueSize = 512;

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
  worker_ = std::thread([this]() { worker_loop(); });
}

WandbLogger::~WandbLogger() {
  finish();
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

void WandbLogger::add_metric(const std::string& section, const std::string& key, nlohmann::json value) {
  if (!enabled_) return;
  if (!section.empty() && !key.empty()) {
    sections_[key] = section;
  }
  payload_[key] = std::move(value);
}

void WandbLogger::add_table(
    const std::string& key,
    const std::vector<std::string>& columns,
    const std::vector<std::vector<nlohmann::json>>& data) {
  if (!enabled_ || key.empty()) return;
  nlohmann::json table = nlohmann::json::object();
  table["_type"] = "table";
  table["columns"] = columns;
  table["data"] = data;

  // Tables render as native wandb.Table widgets in the run's Tables tab.
  // Do NOT register them in sections_: the workspace organizer in
  // wandb_stream.py would otherwise create a phantom line-plot panel for the
  // key, and wandb would render the non-numeric table values as NaN.
  payload_[key] = std::move(table);
}

void WandbLogger::commit(std::int64_t global_step) {
  if (!enabled_) return;
  payload_["_step"] = global_step;
  payload_["_wandb_sections"] = sections_;
  payload_["_wandb_section_order"] = nlohmann::json::array({
      "Tables",
      "1v1",
      "2v2",
      "3v3",
      "Rewards",
      "GCRL",
      "Predictor",
      "Exploration",
      "ES-LoRA",
      "Optimization",
      "Charts",
      "System",
      "Hidden Panels"
  });

  log(payload_);

  // Reset buffers for the next step
  payload_ = nlohmann::json::object();
  sections_ = nlohmann::json::object();
}

void WandbLogger::log(const nlohmann::json& payload) {
  if (!enabled_) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || worker_failed_) {
      enabled_ = false;
      return;
    }
    if (queue_.size() >= kMaxQueueSize) {
      // Subprocess is stalled — drop the oldest entry to bound memory usage.
      queue_.pop_front();
    }
    queue_.push_back(payload.dump());
  }
  cv_.notify_one();
}

void WandbLogger::worker_loop() {
  for (;;) {
    std::string line;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) {
        if (stopping_) {
          return;
        }
        continue;
      }
      line = std::move(queue_.front());
      queue_.pop_front();
    }

    const std::size_t written = std::fwrite(line.data(), 1, line.size(), pipe_);
    if (written != line.size() || std::fwrite("\n", 1, 1, pipe_) != 1 || std::fflush(pipe_) != 0) {
      std::cerr << "wandb logging pipe write failed (errno " << errno << "); disabling further logging\n";
      std::lock_guard<std::mutex> lock(mutex_);
      worker_failed_ = true;
      queue_.clear();
      return;
    }
  }
}

void WandbLogger::finish() {
  if (worker_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_one();
    worker_.join();
  }
  if (pipe_ == nullptr) {
    enabled_ = false;
    return;
  }
  const int status = pclose(pipe_);
  pipe_ = nullptr;
  enabled_ = false;
  if (status != 0) {
    std::cerr << "wandb logging process exited with status " << status << '\n';
  }
}

}  // namespace pulsar
