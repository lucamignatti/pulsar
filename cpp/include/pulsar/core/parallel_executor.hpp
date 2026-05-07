#pragma once

#include <algorithm>
#include <atomic>
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

    {
      std::lock_guard<std::mutex> lock(mutex_);
      count_ = count;
      chunk_size_ = compute_chunk_size(count);
      next_index_.store(0, std::memory_order_relaxed);
      pending_workers_ = workers_.size();
      task_ = std::function<void(std::size_t, std::size_t)>(std::forward<Fn>(fn));
      ++generation_;
    }
    start_cv_.notify_all();

    run_chunks(count, chunk_size_, task_);

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

  std::size_t compute_chunk_size(std::size_t count) const {
    if (worker_count_ <= 1) {
      return count;
    }
    return std::max<std::size_t>(1, count / (worker_count_ * 8));
  }

  void worker_loop(std::size_t worker_index) {
#if defined(PULSAR_ENABLE_TRACING)
    const std::string thread_name = "parallel_executor_" + std::to_string(worker_index);
    PULSAR_TRACE_SET_THREAD_NAME(thread_name);
#endif
    std::size_t local_generation = 0;
    while (true) {
      std::function<void(std::size_t, std::size_t)> task;
      std::size_t count = 0;
      std::size_t chunk_size = 1;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [this, local_generation]() { return stop_ || generation_ != local_generation; });
        if (stop_) {
          return;
        }
        local_generation = generation_;
        task = task_;
        count = count_;
        chunk_size = chunk_size_;
      }

      try {
        run_chunks(count, chunk_size, task);
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (pending_workers_ > 0) {
            --pending_workers_;
            if (pending_workers_ == 0) {
              done_cv_.notify_one();
            }
          }
        }
        throw;
      }

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

  void run_chunks(
      std::size_t count,
      std::size_t chunk_size,
      const std::function<void(std::size_t, std::size_t)>& fn) const {
    while (true) {
      const std::size_t begin = next_index_.fetch_add(chunk_size, std::memory_order_relaxed);
      if (begin >= count) {
        return;
      }
      const std::size_t end = std::min(count, begin + chunk_size);
      fn(begin, end);
    }
  }

  std::size_t worker_count_ = 1;
  std::vector<std::thread> workers_{};
  std::mutex mutex_{};
  std::condition_variable start_cv_{};
  std::condition_variable done_cv_{};
  std::function<void(std::size_t, std::size_t)> task_{};
  std::size_t count_ = 0;
  std::size_t chunk_size_ = 1;
  std::size_t pending_workers_ = 0;
  std::size_t generation_ = 0;
  bool stop_ = false;
  mutable std::atomic<std::size_t> next_index_{0};
};

}  // namespace pulsar
