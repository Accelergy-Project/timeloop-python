#include "pytimeloop/model/accelerator-pool.h"

AcceleratorPool::AcceleratorPool(const model::Engine::Specs& arch_specs,
                                 unsigned num_workers)
    : arch_specs_(arch_specs),
      terminate_(false),
      cur_id_(0),
      task_q_(num_workers),
      result_q_(num_workers) {
  for (int i = 0; i < num_workers; i++) {
    workers_.push_back(std::thread(&AcceleratorPool::worker_loop, this, i));
  }
};

AcceleratorPool::~AcceleratorPool() { Terminate(); }

uint64_t AcceleratorPool::Evaluate(
    Mapping mapping, const problem::Workload& workload,
    const sparse::SparseOptimizationInfo& sparse_optimizations,
    bool break_on_failure) {
  uint64_t task_id = cur_id_.fetch_add(1);

  // Each new evaluate increments task_id, use it to balance queues
  int i = task_id % workers_.size();

  while (!task_q_.at(i).push(EvaluationTask{
      task_id, mapping, workload, sparse_optimizations, break_on_failure})) {
  }

  return task_id;
}

EvaluationResult AcceleratorPool::GetResult() {
  while (true) {
    EvaluationResult result;
    for (int i = 0; i < workers_.size(); i++) {
      bool success = result_q_.at(i).pop(result);
      if (success) {
        return result;
      }
    }
    std::this_thread::yield();
  }
}

void AcceleratorPool::Terminate() {
  terminate_.store(true);

  for (int i = 0; i < workers_.size(); i++) {
    workers_[i].join();
  }

  workers_.clear();
}

void AcceleratorPool::worker_loop(int i) {
  model::Engine engine;
  engine.Spec(arch_specs_);
  auto level_names = arch_specs_.topology.LevelNames();

  auto& task_q = task_q_[i];

  while (true) {
    // Acquire task from task_q_
    EvaluationTask task;
    task_q.pop(task, [&]() { return terminate_.load(); });

    if (terminate_.load()) {
      break;
    }

    // Perform evaluation
    auto pre_eval_status = engine.PreEvaluationCheck(
        task.mapping, task.workload, &task.sparse_optimizations,
        task.break_on_failure);
    bool success =
        std::accumulate(pre_eval_status.begin(), pre_eval_status.end(), true,
                        [](bool cur, const model::EvalStatus& status) {
                          return cur && status.success;
                        });
    if (!success) {
      queue_result(i, EvaluationResult{task.id, pre_eval_status, std::nullopt});
      continue;
    }

    auto eval_status =
        engine.Evaluate(task.mapping, task.workload, &task.sparse_optimizations,
                        task.break_on_failure);
    for (unsigned level = 0; level < eval_status.size(); level++) {
      if (!eval_status[level].success) {
        std::cerr << "ERROR: couldn't map level " << level_names.at(level)
                  << ": " << eval_status[level].fail_reason << std::endl;
      }
    }

    if (engine.IsEvaluated()) {
      auto topology = engine.GetTopology();
      queue_result(i, EvaluationResult{task.id, pre_eval_status, eval_status,
                                       engine.Utilization(), engine.Energy(),
                                       engine.Area(), engine.Cycles(),
                                       topology.AlgorithmicComputes(),
                                       topology.ActualComputes(),
                                       topology.LastLevelAccesses()});
    } else {
      queue_result(i, EvaluationResult{task.id, pre_eval_status, std::nullopt});
    }
  }
}

void AcceleratorPool::queue_result(int i, EvaluationResult eval_result) {
  while (!result_q_[i].push(eval_result)) {
    std::this_thread::yield();
  }
}
