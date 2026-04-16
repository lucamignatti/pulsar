#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

namespace pulsar::tracing {

namespace detail {

struct TraceRecorder;

TraceRecorder* current_recorder() noexcept;
std::uint32_t current_thread_id() noexcept;
std::uint64_t now_microseconds(TraceRecorder* recorder) noexcept;
void record_process_name(TraceRecorder* recorder, std::string_view process_name) noexcept;
void record_thread_name(
    TraceRecorder* recorder,
    std::string_view thread_name,
    std::uint32_t thread_id) noexcept;
void record_complete_event(
    TraceRecorder* recorder,
    std::string_view category,
    std::string_view name,
    std::uint32_t thread_id,
    std::uint64_t start_us,
    std::uint64_t duration_us) noexcept;

}  // namespace detail

class Session {
 public:
  Session(const std::filesystem::path& output_path, std::string_view process_name);
  ~Session();

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;

  [[nodiscard]] bool enabled() const noexcept;

 private:
  std::unique_ptr<detail::TraceRecorder> recorder_{};
  detail::TraceRecorder* previous_recorder_ = nullptr;
};

class Scope {
 public:
  Scope(std::string_view category, std::string_view name) noexcept;
  ~Scope();

  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  detail::TraceRecorder* recorder_ = nullptr;
  std::uint32_t thread_id_ = 0;
  std::uint64_t start_us_ = 0;
  std::string_view category_{};
  std::string_view name_{};
};

void set_thread_name(std::string_view name) noexcept;

}  // namespace pulsar::tracing

#if defined(PULSAR_ENABLE_TRACING)

#define PULSAR_TRACE_CONCAT_INNER(a, b) a##b
#define PULSAR_TRACE_CONCAT(a, b) PULSAR_TRACE_CONCAT_INNER(a, b)
#define PULSAR_TRACE_SCOPE(name) \
  ::pulsar::tracing::Scope PULSAR_TRACE_CONCAT(_pulsar_trace_scope_, __LINE__)("", (name))
#define PULSAR_TRACE_SCOPE_CAT(category, name) \
  ::pulsar::tracing::Scope PULSAR_TRACE_CONCAT(_pulsar_trace_scope_, __LINE__)((category), (name))
#define PULSAR_TRACE_FUNCTION() PULSAR_TRACE_SCOPE(__func__)
#define PULSAR_TRACE_SET_THREAD_NAME(name) ::pulsar::tracing::set_thread_name((name))

#else

#define PULSAR_TRACE_SCOPE(name) ((void)0)
#define PULSAR_TRACE_SCOPE_CAT(category, name) ((void)0)
#define PULSAR_TRACE_FUNCTION() ((void)0)
#define PULSAR_TRACE_SET_THREAD_NAME(name) ((void)0)

#endif
