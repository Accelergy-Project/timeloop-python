#include "pytimeloop/model/accelerator.h"

namespace pytimeloop::pymodel {

Accelerator::Accelerator(const model::Engine::Specs& arch_specs)
    : arch_specs_(arch_specs) {
  engine_.Spec(arch_specs_);
  level_names_ = arch_specs_.topology.LevelNames();
};

EvaluationResult Accelerator::Evaluate(
    Mapping mapping, problem::Workload& workload,
    sparse::SparseOptimizationInfo& sparse_optimizations,
    bool break_on_failure) {
  auto pre_eval_status = engine_.PreEvaluationCheck(
      mapping, workload, &sparse_optimizations, break_on_failure);
  bool success =
      std::accumulate(pre_eval_status.begin(), pre_eval_status.end(), true,
                      [](bool cur, const model::EvalStatus& status) {
                        return cur && status.success;
                      });
  if (!success) {
    return EvaluationResult::FailedEvaluation(pre_eval_status);
  }

  auto eval_status = engine_.Evaluate(mapping, workload, &sparse_optimizations,
                                      break_on_failure);

  if (engine_.IsEvaluated()) {
    auto topology = engine_.GetTopology();
    return EvaluationResult{0,
                            pre_eval_status,
                            eval_status,
                            engine_.Utilization(),
                            engine_.Energy(),
                            engine_.Area(),
                            engine_.Cycles(),
                            topology.AlgorithmicComputes(),
                            topology.ActualComputes(),
                            topology.LastLevelAccesses()};
  } else {
    return EvaluationResult::FailedEvaluation(pre_eval_status);
  }
}

}  // namespace pytimeloop::pymodel
