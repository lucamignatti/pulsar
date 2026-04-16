#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pulsar/tracing/tracing.hpp"

namespace pulsar {

class ParallelExecutor {
 public:
  explicit ParallelExecutor(std::size_t worker_count = 0) {
    worker_count_ = resolve_worker_count(worker_count);
    if (worker_count_ <= 1) {
      return;
    }

    workers_.reserve(worker_count_ - 1);
    for (std::size_t worker_index = 0; worker_index + 1 < worker_count_; ++worker_index) {
      workers_.emplace_back([this, worker_index]() { worker_loop(worker_index); });
    }
  }

  ParallelExecutor(const ParallelExecutor&) = delete;
  ParallelExecutor& operator=(const ParallelExecutor&) = delete;

  ~ParallelExecutor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    start_cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  [[nodiscard]] std::size_t worker_count() const {
    return worker_count_;
  }

  template <typename Fn>
  void parallel_for(std::size_t count, Fn&& fn) {
    if (count == 0) {
      return;
    }
    if (worker_count_ <= 1 || count == 1) {
      fn(0, count);
      return;
    }

    std::function<void(std::size_t, std::size_t)> task(std::forward<Fn>(fn));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      count_ = count;
      pending_workers_ = workers_.size();
      task_ = &task;
      ++generation_;
    }
    start_cv_.notify_all();

    run_chunk(worker_count_ - 1, count, task);

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return pending_workers_ == 0; });
    task_ = nullptr;
  }

 private:
  static std::size_t resolve_worker_count(std::size_t requested) {
    if (requested != 0) {
      return std::max<std::size_t>(1, requested);
    }
    const unsigned int detected = std::thread::hardware_concurrency();
    return std::max<std::size_t>(1, detected == 0 ? 1U : detected);
  }

  void worker_loop(std::size_t worker_index) {
#if defined(PULSAR_ENABLE_TRACING)
    const std::string thread_name = "parallel_executor_" + std::to_string(worker_index);
    PULSAR_TRACE_SET_THREAD_NAME(thread_name);
#endif
    std::size_t local_generation = 0;
    while (true) {
      const std::function<void(std::size_t, std::size_t)>* task = nullptr;
      std::size_t count = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [this, local_generation]() { return stop_ || generation_ != local_generation; });
        if (stop_) {
          return;
        }
        local_generation = generation_;
        task = task_;
        count = count_;
      }

      run_chunk(worker_index, count, *task);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_workers_ > 0) {
          --pending_workers_;
          if (pending_workers_ == 0) {
            done_cv_.notify_one();
          }
        }
      }
    }
  }

  void run_chunk(
      std::size_t worker_index,
      std::size_t count,
      const std::function<void(std::size_t, std::size_t)>& fn) const {
    const std::size_t begin = (count * worker_index) / worker_count_;
    const std::size_t end = (count * (worker_index + 1)) / worker_count_;
    if (begin < end) {
      fn(begin, end);
    }
  }

  std::size_t worker_count_ = 1;
  std::vector<std::thread> workers_{};
  std::mutex mutex_{};
  std::condition_variable start_cv_{};
  std::condition_variable done_cv_{};
  const std::function<void(std::size_t, std::size_t)>* task_ = nullptr;
  std::size_t count_ = 0;
  std::size_t pending_workers_ = 0;
  std::size_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace pulsar
