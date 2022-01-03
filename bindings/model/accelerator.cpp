#include "bindings/model/accelerator.h"

#include "bindings/model/bindings.h"

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
    return EvaluationResult{.pre_eval_status = pre_eval_status,
                            .eval_status = std::nullopt};
  }

  auto eval_status = engine_.Evaluate(mapping, workload, &sparse_optimizations,
                                      break_on_failure);
  for (unsigned level = 0; level < eval_status.size(); level++) {
    if (!eval_status[level].success) {
      std::cerr << "ERROR: couldn't map level " << level_names_.at(level)
                << ": " << eval_status[level].fail_reason << std::endl;
    }
  }

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
    return EvaluationResult{0, pre_eval_status, std::nullopt};
  }
}

namespace model_bindings {

void BindAccelerator(py::module& m) {
  py::class_<Accelerator>(m, "NativeAccelerator")
      .def(py::init<const model::Engine::Specs&>())
      .def("evaluate", &Accelerator::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>());
}

}  // namespace model_bindings
