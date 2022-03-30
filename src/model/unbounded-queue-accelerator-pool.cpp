#include "pytimeloop/model/accelerator-pool.h"
#include "pytimeloop/model/accelerator.h"

namespace pytimeloop::pymodel {

UnboundedQueueAcceleratorPool::UnboundedQueueAcceleratorPool(
    const model::Engine::Specs& arch_specs, unsigned num_workers)
    : arch_specs_(arch_specs),
      terminate_(false),
      cur_id_(0),
      task_q_(num_workers),
      result_q_(num_workers) {
  for (unsigned i = 0; i < num_workers; i++) {
    workers_.push_back(
        std::thread(&UnboundedQueueAcceleratorPool::worker_loop, this, i));
  }
};

UnboundedQueueAcceleratorPool::~UnboundedQueueAcceleratorPool() { Terminate(); }

uint64_t UnboundedQueueAcceleratorPool::Evaluate(
    Mapping mapping, const problem::Workload& workload,
    const sparse::SparseOptimizationInfo& sparse_optimizations,
    bool break_on_failure) {
  uint64_t task_id = cur_id_.fetch_add(1);

  // Each new evaluate increments task_id, use it to balance queues
  int i = task_id % workers_.size();

  while (!task_q_.at(i).Push(EvaluationTask{
      task_id, mapping, workload, sparse_optimizations, break_on_failure})) {
  }

  return task_id;
}

EvaluationResult UnboundedQueueAcceleratorPool::GetResult() {
  EvaluationResult result;
  while (true) {
    for (unsigned i = 0; i < workers_.size(); i++) {
      bool success = result_q_.at(i).Pop(result);
      if (success) {
        return result;
      }
    }
  }
}

void UnboundedQueueAcceleratorPool::Terminate() {
  terminate_.store(true);
  for (auto& q : task_q_) {
    q.NotifyAll();
  }
  for (auto& q : result_q_) {
    q.NotifyAll();
  }

  for (unsigned i = 0; i < workers_.size(); i++) {
    workers_[i].join();
  }

  workers_.clear();
}

void UnboundedQueueAcceleratorPool::worker_loop(int i) {
  Accelerator acc(arch_specs_);

  auto& task_q = task_q_[i];

  while (!terminate_) {
    // Acquire task from task_q_
    EvaluationTask task;
    task_q.Pop(task, [&]() { return terminate_.load(); });

    if (terminate_.load()) {
      break;
    }

    auto res = acc.Evaluate(task.mapping, task.workload,
                            task.sparse_optimizations, task.break_on_failure);
    res.id = task.id;
    queue_result(i, std::move(res));
  }
}

void UnboundedQueueAcceleratorPool::queue_result(
    int i, const EvaluationResult& eval_result) {
  while (!result_q_[i].Push(eval_result)) {
    std::this_thread::yield();
  }
}

}  // namespace pytimeloop::pymodel
