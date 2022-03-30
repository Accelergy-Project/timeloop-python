#include "pytimeloop/model/accelerator-pool.h"
#include "pytimeloop/model/accelerator.h"

namespace pytimeloop::pymodel {

BoundedQueueAcceleratorPool::BoundedQueueAcceleratorPool(
    const model::Engine::Specs& arch_specs, size_t num_workers,
    size_t num_threads)
    : arch_specs_(arch_specs),
      terminate_(false),
      num_workers_(num_workers),
      num_threads_(num_threads),
      workers_(num_workers) {
  for (unsigned i = 0; i < num_threads; i++) {
    threads_.emplace_back(&BoundedQueueAcceleratorPool::WorkerLoop, this, i);
  }
  assert(WorkersAvailable() == num_workers);
}

BoundedQueueAcceleratorPool::~BoundedQueueAcceleratorPool() { Terminate(); }

uint64_t BoundedQueueAcceleratorPool::Evaluate(
    Mapping mapping, const problem::Workload& workload,
    const sparse::SparseOptimizationInfo& sparse_opts, bool break_on_failure) {
  for (auto& w : workers_) {
    if (w.idle.load()) {
      uint64_t task_id = w.idx++;

      w.task.id = task_id;
      w.task.mapping = std::move(mapping);
      w.task.workload = workload;
      w.task.sparse_optimizations = sparse_opts;
      w.task.break_on_failure = break_on_failure;

      w.started = true;
      w.idle.store(false);

      return task_id;
    }
  }
  assert(false);
}

size_t BoundedQueueAcceleratorPool::WorkersAvailable() const {
  auto idle_workers = 0;
  for (auto& w : workers_) {
    idle_workers += w.idle.load();
  }
  return idle_workers;
}

EvaluationResult BoundedQueueAcceleratorPool::GetResult() {
  while (true) {
    for (auto& w : workers_) {
      if (w.idle.load() && w.started) {
        return std::move(w.res);
      }
    }
  }
}

void BoundedQueueAcceleratorPool::Terminate() {
  terminate_.store(true);

  for (auto& t : threads_) {
    t.join();
  }
}

void BoundedQueueAcceleratorPool::WorkerLoop(size_t i) {
  Accelerator accelerator(arch_specs_);

  while (true) {
    while (workers_.at(i).idle.load() && !terminate_.load()) {
      i = (i + num_threads_) % num_workers_;
    }
    if (terminate_.load()) {
      return;
    }

    auto& worker = workers_.at(i);

    worker.res = accelerator.Evaluate(
        std::move(worker.task.mapping), worker.task.workload,
        worker.task.sparse_optimizations, worker.task.break_on_failure);

    worker.idle.store(true);
    i = (i + num_threads_) % num_workers_;
  }
}

}  // namespace pytimeloop::pymodel
