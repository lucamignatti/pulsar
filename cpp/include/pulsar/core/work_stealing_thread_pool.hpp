#pragma once

#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace pulsar {

class WorkStealingThreadPool {
 public:
  explicit WorkStealingThreadPool(std::size_t worker_count = 0)
      : next_queue_(0), stop_(false) {
    if (worker_count == 0) {
      worker_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    queues_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
      queues_.push_back(std::make_unique<TaskQueue>());
    }
    workers_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers_.emplace_back([this, i]() { worker_loop(i); });
    }
  }

  WorkStealingThreadPool(const WorkStealingThreadPool&) = delete;
  WorkStealingThreadPool& operator=(const WorkStealingThreadPool&) = delete;

  ~WorkStealingThreadPool() {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  template <typename Fn>
  auto submit(Fn&& fn) -> std::future<std::invoke_result_t<Fn>> {
    using Result = std::invoke_result_t<Fn>;
    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Fn>(fn));
    std::future<Result> future = task->get_future();
    const std::size_t queue = next_queue_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
    {
      std::lock_guard<std::mutex> lock(queues_[queue]->mutex);
      queues_[queue]->tasks.emplace_back([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

 private:
  struct TaskQueue {
    std::mutex mutex;
    std::deque<std::function<void()>> tasks;
  };

  bool pop_local(std::size_t index, std::function<void()>& task) {
    std::lock_guard<std::mutex> lock(queues_[index]->mutex);
    if (queues_[index]->tasks.empty()) {
      return false;
    }
    task = std::move(queues_[index]->tasks.back());
    queues_[index]->tasks.pop_back();
    return true;
  }

  bool steal(std::size_t index, std::function<void()>& task) {
    for (std::size_t i = 0; i < queues_.size(); ++i) {
      const std::size_t victim = (index + 1 + i) % queues_.size();
      std::lock_guard<std::mutex> lock(queues_[victim]->mutex);
      if (!queues_[victim]->tasks.empty()) {
        task = std::move(queues_[victim]->tasks.front());
        queues_[victim]->tasks.pop_front();
        return true;
      }
    }
    return false;
  }

  bool has_work() {
    for (auto& queue : queues_) {
      std::lock_guard<std::mutex> lock(queue->mutex);
      if (!queue->tasks.empty()) {
        return true;
      }
    }
    return false;
  }

  void worker_loop(std::size_t index) {
    while (!stop_.load(std::memory_order_acquire)) {
      std::function<void()> task;
      if (pop_local(index, task) || steal(index, task)) {
        task();
        continue;
      }
      std::unique_lock<std::mutex> lock(cv_mutex_);
      cv_.wait(lock, [this]() {
        return stop_.load(std::memory_order_acquire) || has_work();
      });
    }
  }

  std::vector<std::unique_ptr<TaskQueue>> queues_;
  std::vector<std::thread> workers_;
  std::atomic<std::size_t> next_queue_;
  std::atomic<bool> stop_;
  std::mutex cv_mutex_;
  std::condition_variable cv_;
};

}  // namespace pulsar
