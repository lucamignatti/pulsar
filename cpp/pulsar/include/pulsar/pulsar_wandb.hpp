#pragma once

#include <cstdio>
#include <string>

namespace pulsar {

// Minimal WandB logger: spawns scripts/wandb_stream.py as a subprocess and
// pipes newline-delimited JSON metric objects to it.
//
// Usage:
//   PulsarWandB wb;
//   if (wb.init("pulsar", "run-name")) { ... }
//   wb.log("{\"loss\":0.1,\"step\":1}");
class PulsarWandB {
 public:
  PulsarWandB() = default;
  ~PulsarWandB() { close(); }

  PulsarWandB(const PulsarWandB&) = delete;
  PulsarWandB& operator=(const PulsarWandB&) = delete;

  // Spawn the stream script. Returns false if popen fails (wandb not installed, etc.).
  bool init(const std::string& project,
            const std::string& run_name = "",
            const std::string& entity   = "",
            const std::string& python   = "python3",
            const std::string& script   = "scripts/wandb_stream.py") {
    if (project.empty()) return false;
    std::string cmd = python + " " + script
                    + " " + sq(project)
                    + " " + sq(run_name)
                    + " " + sq(entity)
                    + " 2>/dev/null";
    pipe_ = popen(cmd.c_str(), "w");
    return pipe_ != nullptr;
  }

  // Write a pre-built JSON object string (no trailing newline needed).
  void log(const std::string& json_object) {
    if (!pipe_) return;
    std::fputs((json_object + "\n").c_str(), pipe_);
    std::fflush(pipe_);
  }

  // Convenience: log step + named float metrics.
  void log(int step,
           std::initializer_list<std::pair<const char*, double>> metrics) {
    if (!pipe_) return;
    char buf[4096];
    int pos = std::snprintf(buf, sizeof(buf), "{\"_step\":%d", step);
    for (const auto& [k, v] : metrics) {
      pos += std::snprintf(buf + pos, sizeof(buf) - static_cast<std::size_t>(pos),
                           ",\"%s\":%.6g", k, v);
      if (pos >= static_cast<int>(sizeof(buf)) - 64) break;
    }
    std::snprintf(buf + pos, sizeof(buf) - static_cast<std::size_t>(pos), "}\n");
    std::fputs(buf, pipe_);
    std::fflush(pipe_);
  }

  void close() {
    if (pipe_) {
      pclose(pipe_);
      pipe_ = nullptr;
    }
  }

  bool is_active() const { return pipe_ != nullptr; }

 private:
  static std::string sq(const std::string& s) {
    std::string out = "'";
    for (char c : s) { if (c == '\'') out += "'\\''"; else out += c; }
    return out + "'";
  }

  FILE* pipe_ = nullptr;
};

}  // namespace pulsar
