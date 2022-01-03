#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include "bindings/model/eval-result.h"

// PyBind11
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

// Timeloop headers
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <model/sparse-optimization-info.hpp>
#include <workload/workload.hpp>

namespace py = pybind11;

struct EvaluationTask {
  uint64_t id;
  Mapping mapping;
  problem::Workload workload;
  sparse::SparseOptimizationInfo sparse_optimizations;
  bool break_on_failure;
  bool auto_bypass_on_failure;
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
