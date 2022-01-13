#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "pytimeloop/model/eval-result.h"

// Timeloop headers
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <model/sparse-optimization-info.hpp>
#include <workload/workload.hpp>

const size_t CACHE_LINE = 64;

struct EvaluationTask {
  uint64_t id;
  Mapping mapping;
  problem::Workload workload;
  sparse::SparseOptimizationInfo sparse_optimizations;
  bool break_on_failure;
};

template <typename T>
struct Aligned {
  T data;
  std::array<char, ((sizeof(T) - 1) / CACHE_LINE + 1) * CACHE_LINE> padding;

  Aligned() {}
  Aligned(const T& val) : data(val) {}
};

template <typename T>
class SpscQueue {
 public:
  SpscQueue(size_t capacity)
      : cached_head_(0),
        cached_tail_(0),
        head_(0),
        tail_(0),
        capacity_(capacity) {
    arr_ = std::make_unique<Aligned<T>[]>(capacity);
  }

  bool push(const T& val) {
    auto h = head_.load(std::memory_order_relaxed);

    if (((h + 1) % capacity_) == cached_tail_) {
      auto t = tail_.load(std::memory_order_acquire);
      if (t == cached_tail_) {
        return false;
      }
      cached_tail_ = t;
    }
    arr_[h] = Aligned(val);
    head_.store((h + 1) % capacity_, std::memory_order_release);
    return true;
  }

  bool pop(T& val) {
    auto t = tail_.load(std::memory_order_relaxed);

    if (t == cached_head_) {
      auto h = head_.load(std::memory_order_acquire);
      if (h == cached_head_) {
        return false;
      }
      cached_head_ = h;
    }
    val = arr_[t].data;
    tail_.store((t + 1) % capacity_, std::memory_order_release);
    return true;
  }

 private:
  alignas(CACHE_LINE) std::unique_ptr<Aligned<T>[]> arr_;

  alignas(CACHE_LINE) std::atomic_size_t head_;
  size_t cached_tail_;

  alignas(CACHE_LINE) std::atomic_size_t tail_;

  alignas(CACHE_LINE) size_t cached_head_;

  const size_t capacity_;
};

template <typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() {}

  bool push(const T& val) {
    std::lock_guard<std::mutex> enq_lock(m_);
    q_.push(val);
    cv_.notify_all();
    return true;
  }

  bool pop(T& val) {
    std::unique_lock<std::mutex> deq_lock(m_);
    if (q_.empty()) {
      return false;
    }
    val = std::move(q_.front());
    q_.pop();
    return true;
  }

  template <typename Predicate>
  bool pop(T& val, Predicate stop_waiting) {
    std::unique_lock<std::mutex> deq_lock(m_);
    while (q_.empty() && !stop_waiting()) {
      cv_.wait(deq_lock);
    }
    if (q_.empty()) {
      return false;
    }
    val = std::move(q_.front());
    q_.pop();
    return true;
  }

 private:
  std::queue<T> q_;
  std::mutex m_;
  std::condition_variable cv_;
};

class AcceleratorPool {
 public:
  AcceleratorPool(const model::Engine::Specs& arch_specs, unsigned num_workers);

  ~AcceleratorPool();

  uint64_t Evaluate(Mapping mapping, const problem::Workload& workload,
                    const sparse::SparseOptimizationInfo& sparse_optimizations,
                    bool break_on_failure = false);

  EvaluationResult GetResult();

  void Terminate();

 private:
  const model::Engine::Specs& arch_specs_;

  std::vector<std::thread> workers_;
  std::atomic<bool> terminate_;

  std::atomic<uint64_t> cur_id_;

  std::vector<ConcurrentQueue<EvaluationTask>> task_q_;
  std::vector<ConcurrentQueue<EvaluationResult>> result_q_;

  void worker_loop(int i);

  void queue_result(int i, EvaluationResult eval_result);
};
