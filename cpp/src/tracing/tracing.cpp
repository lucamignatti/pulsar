#include "pulsar/tracing/tracing.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace pulsar::tracing {
namespace {

using Clock = std::chrono::steady_clock;

std::atomic<detail::TraceRecorder*> g_active_recorder{nullptr};
std::atomic<std::uint32_t> g_next_thread_id{1};
thread_local std::uint32_t g_thread_id = 0;

std::int32_t current_process_id() noexcept {
#if defined(_WIN32)
  return _getpid();
#else
  return static_cast<std::int32_t>(getpid());
#endif
}

std::string escape_json(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          std::ostringstream hex;
          hex << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(ch));
          escaped += hex.str();
        } else {
          escaped.push_back(ch);
        }
        break;
    }
  }
  return escaped;
}

void append_json_event(detail::TraceRecorder* recorder, std::string event) noexcept;

}  // namespace

namespace detail {

struct TraceRecorder {
  explicit TraceRecorder(std::filesystem::path path)
      : output_path(std::move(path)),
        start_time(Clock::now()),
        process_id(current_process_id()) {}

  std::filesystem::path output_path{};
  std::ofstream stream{};
  std::mutex mutex{};
  Clock::time_point start_time{};
  std::int32_t process_id = 0;
  bool first_event = true;
};

TraceRecorder* current_recorder() noexcept {
  return g_active_recorder.load(std::memory_order_acquire);
}

std::uint32_t current_thread_id() noexcept {
  if (g_thread_id == 0) {
    g_thread_id = g_next_thread_id.fetch_add(1, std::memory_order_relaxed);
  }
  return g_thread_id;
}

std::uint64_t now_microseconds(TraceRecorder* recorder) noexcept {
  if (recorder == nullptr) {
    return 0;
  }
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - recorder->start_time).count());
}

void record_process_name(TraceRecorder* recorder, std::string_view process_name) noexcept {
  if (recorder == nullptr) {
    return;
  }
  std::ostringstream event;
  event << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << recorder->process_id
        << ",\"tid\":0,\"ts\":0,\"args\":{\"name\":\"" << escape_json(process_name) << "\"}}";
  append_json_event(recorder, event.str());
}

void record_thread_name(
    TraceRecorder* recorder,
    std::string_view thread_name,
    std::uint32_t thread_id) noexcept {
  if (recorder == nullptr) {
    return;
  }
  std::ostringstream event;
  event << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" << recorder->process_id
        << ",\"tid\":" << thread_id << ",\"ts\":0,\"args\":{\"name\":\"" << escape_json(thread_name) << "\"}}";
  append_json_event(recorder, event.str());
}

void record_complete_event(
    TraceRecorder* recorder,
    std::string_view category,
    std::string_view name,
    std::uint32_t thread_id,
    std::uint64_t start_us,
    std::uint64_t duration_us) noexcept {
  if (recorder == nullptr) {
    return;
  }
  std::ostringstream event;
  event << "{\"name\":\"" << escape_json(name) << "\",\"cat\":\"" << escape_json(category)
        << "\",\"ph\":\"X\",\"pid\":" << recorder->process_id
        << ",\"tid\":" << thread_id
        << ",\"ts\":" << start_us
        << ",\"dur\":" << duration_us << "}";
  append_json_event(recorder, event.str());
}

}  // namespace detail

namespace {

void append_json_event(detail::TraceRecorder* recorder, std::string event) noexcept {
  if (recorder == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(recorder->mutex);
  if (!recorder->stream.is_open()) {
    return;
  }
  if (!recorder->first_event) {
    recorder->stream << ",\n";
  }
  recorder->first_event = false;
  recorder->stream << event;
}

}  // namespace

Session::Session(const std::filesystem::path& output_path, std::string_view process_name) {
#if defined(PULSAR_ENABLE_TRACING)
  if (output_path.empty()) {
    return;
  }

  try {
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }

    recorder_ = std::make_unique<detail::TraceRecorder>(output_path);
    recorder_->stream.open(output_path, std::ios::out | std::ios::trunc);
    if (!recorder_->stream.is_open()) {
      std::cerr << "pulsar tracing disabled: failed to open " << output_path << '\n';
      recorder_.reset();
      return;
    }
    recorder_->stream << "{\"traceEvents\":[\n";
    previous_recorder_ = g_active_recorder.exchange(recorder_.get(), std::memory_order_acq_rel);
    detail::record_process_name(recorder_.get(), process_name.empty() ? std::string_view("pulsar") : process_name);
  } catch (const std::exception& exc) {
    std::cerr << "pulsar tracing disabled: " << exc.what() << '\n';
    recorder_.reset();
  }
#else
  (void)output_path;
  (void)process_name;
#endif
}

Session::~Session() {
#if defined(PULSAR_ENABLE_TRACING)
  if (!recorder_) {
    return;
  }
  if (g_active_recorder.load(std::memory_order_acquire) == recorder_.get()) {
    g_active_recorder.store(previous_recorder_, std::memory_order_release);
  }
  std::lock_guard<std::mutex> lock(recorder_->mutex);
  if (recorder_->stream.is_open()) {
    recorder_->stream << "\n]}\n";
    recorder_->stream.flush();
  }
#endif
}

bool Session::enabled() const noexcept {
  return recorder_ != nullptr;
}

Scope::Scope(std::string_view category, std::string_view name) noexcept
    : recorder_(detail::current_recorder()),
      category_(category),
      name_(name) {
  if (recorder_ == nullptr) {
    return;
  }
  thread_id_ = detail::current_thread_id();
  start_us_ = detail::now_microseconds(recorder_);
}

Scope::~Scope() {
  if (recorder_ == nullptr) {
    return;
  }
  const std::uint64_t end_us = detail::now_microseconds(recorder_);
  detail::record_complete_event(recorder_, category_, name_, thread_id_, start_us_, end_us - start_us_);
}

void set_thread_name(std::string_view name) noexcept {
#if defined(PULSAR_ENABLE_TRACING)
  detail::TraceRecorder* recorder = detail::current_recorder();
  if (recorder == nullptr) {
    return;
  }
  detail::record_thread_name(recorder, name, detail::current_thread_id());
#else
  (void)name;
#endif
}

}  // namespace pulsar::tracing
