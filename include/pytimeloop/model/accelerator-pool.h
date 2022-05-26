#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

// Timeloop headers
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <model/sparse-optimization-info.hpp>
#include <workload/workload.hpp>

#include "pytimeloop/model/eval-result.h"
#include "pytimeloop/utils/concurrent-queue.h"

namespace pytimeloop::pymodel {

struct EvaluationTask {
  uint64_t id;
  Mapping mapping;
  problem::Workload workload;
  sparse::SparseOptimizationInfo sparse_optimizations;
  bool break_on_failure;
};

class AcceleratorPool {
 public:
  virtual uint64_t Evaluate(
      Mapping mapping, const problem::Workload& workload,
      const sparse::SparseOptimizationInfo& sparse_optimizations,
      bool break_on_failure = false) = 0;

  virtual EvaluationResult GetResult() = 0;

  virtual void Terminate() = 0;
};

class UnboundedQueueAcceleratorPool : public AcceleratorPool {
 public:
  UnboundedQueueAcceleratorPool(const model::Engine::Specs& arch_specs,
                                unsigned num_workers);

  ~UnboundedQueueAcceleratorPool();

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

  void queue_result(int i, const EvaluationResult& eval_result);
};

class BoundedQueueAcceleratorPool : public AcceleratorPool {
 public:
  BoundedQueueAcceleratorPool(const model::Engine::Specs& arch_specs,
                              size_t num_workers, size_t num_threads);

  ~BoundedQueueAcceleratorPool();

  uint64_t Evaluate(Mapping mapping, const problem::Workload& workload,
                    const sparse::SparseOptimizationInfo& sparse_optimizations,
                    bool break_on_failure = false);

  size_t WorkersAvailable() const;

  EvaluationResult GetResult();

  void Terminate();

 private:
  struct Worker {
    unsigned idx;
    EvaluationTask task;
    EvaluationResult res;

    bool started;
    std::atomic_bool idle;

    Worker() : idx(0), task(), res(), started(false), idle(true) {}
  };

  const model::Engine::Specs& arch_specs_;

  std::atomic_bool terminate_;
  const size_t num_workers_;
  const size_t num_threads_;
  std::vector<Worker> workers_;
  std::vector<std::thread> threads_;

  void WorkerLoop(size_t i);
};

}  // namespace pytimeloop::pymodel
