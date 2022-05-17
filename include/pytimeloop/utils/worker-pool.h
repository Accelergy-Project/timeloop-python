#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "pytimeloop/utils/concurrent-queue.h"

template <typename T>
struct WithId {
  uint64_t id;
  T val;
};

template <typename Worker>
class WorkerPool {
 public:
  using Task = typename Worker::Task;
  using Result = typename Worker::Result;

 public:
  template <typename WorkerFactory>
  WorkerPool(int n_workers, WorkerFactory&& factory)
      : terminated_(false), task_q_(), result_q_() {
    for (int i = 0; i < n_workers; ++i) {
      threads_.push_back(
          std::thread(&WorkerPool<Worker>::worker_loop, this, factory()));
    }
  }

  uint64_t PushTask(Task& task) {
    auto task_id = cur_id_.fetch_add(1);
    task_q_.Push(WithId<Task>{.id = task_id, .val = task});
    task_q_.NotifyAll();
    return task_id;
  }
  uint64_t PushTask(Task&& task) {
    auto task_id = cur_id_.fetch_add(1);
    task_q_.Push(WithId<Task>{.id = task_id, .val = std::move(task)});
    task_q_.NotifyAll();
    return task_id;
  }

  WithId<Result> PopResult() {
    WithId<Result> result;
    result_q_.Pop(result, [&]() { return terminated_.load(); });
    return result;
  }

  void Terminate() {
    terminated_.store(true);

    task_q_.NotifyAll();
    result_q_.NotifyAll();

    for (auto& t : threads_) {
      t.join();
    }

    threads_.clear();
  }

 private:
  std::vector<std::thread> threads_;
  std::atomic_bool terminated_;
  std::atomic<uint64_t> cur_id_;

  ConcurrentQueue<WithId<Task>> task_q_;
  ConcurrentQueue<WithId<Result>> result_q_;

 private:
  void worker_loop(Worker&& worker) {
    while (!terminated_) {
      WithId<Task> task;
      task_q_.Pop(task, [&]() { return terminated_.load(); });

      if (terminated_) {
        break;
      }

      auto result = worker(task.val);

      result_q_.Push(WithId<Result>{.id = task.id, .val = std::move(result)});
    }
  }
};