#include "bindings/model/eval-result.h"

namespace model_bindings {
void BindEvaluationResult(py::module& m) {
  py::class_<EvaluationResult>(m, "EvaluationResult")
      .def_readonly("id", &EvaluationResult::id)
      .def_readonly("pre_eval_status", &EvaluationResult::pre_eval_status)
      .def_readonly("eval_status", &EvaluationResult::eval_status)
      .def_readonly("utilization", &EvaluationResult::utilization)
      .def_readonly("energy", &EvaluationResult::energy)
      .def_readonly("area", &EvaluationResult::area)
      .def_readonly("cycles", &EvaluationResult::cycles)
      .def_readonly("algorithmic_computes",
                    &EvaluationResult::algorithmic_computes)
      .def_readonly("actual_computes", &EvaluationResult::actual_computes)
      .def_readonly("last_level_accesses",
                    &EvaluationResult::last_level_accesses);
}
}  // namespace model_bindings