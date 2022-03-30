#include "pytimeloop/bindings/model.h"

#include "pytimeloop/model/accelerator-pool.h"
#include "pytimeloop/model/accelerator.h"

// PyBind11 headers
#include <pybind11/stl.h>

// Timeloop headers
#include "model/engine.hpp"
#include "model/level.hpp"
#include "model/sparse-optimization-info.hpp"
#include "model/sparse-optimization-parser.hpp"

namespace pytimeloop::model_bindings {
using namespace pytimeloop::pymodel;

void BindAccelerator(py::module& m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init<const model::Engine::Specs&>())
      .def("evaluate", &Accelerator::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>());
}

void BindAcceleratorPool(py::module& m) {
  py::class_<UnboundedQueueAcceleratorPool>(m, "UnboundedAcceleratorPool")
      .def(py::init<const model::Engine::Specs&, unsigned>())
      .def("evaluate", &UnboundedQueueAcceleratorPool::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("get_result", &UnboundedQueueAcceleratorPool::GetResult);

  py::class_<BoundedQueueAcceleratorPool>(m, "BoundedAcceleratorPool")
      .def(py::init<const model::Engine::Specs&, size_t, size_t>())
      .def("evaluate", &BoundedQueueAcceleratorPool::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("get_result", &BoundedQueueAcceleratorPool::GetResult);
}

void BindEngine(py::module& m) {
  py::class_<model::Engine::Specs>(m, "ArchSpecs")
      .def(py::init(&model::Engine::ParseSpecs))
      .def_static("parse_specs", &model::Engine::ParseSpecs,
                  "Parse architecture specifications.")
      .def("parse_accelergy_art",
           [](model::Engine::Specs& s, config::CompoundConfigNode& art) {
             s.topology.ParseAccelergyART(art);
           })
      .def(
          "parse_accelergy_ert",
          [](model::Engine::Specs& s, config::CompoundConfigNode& ert) {
            s.topology.ParseAccelergyERT(ert);
          },
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>())
      .def("level_names",
           [](model::Engine::Specs& s) { return s.topology.LevelNames(); })
      .def("storage_level_names", [](model::Engine::Specs& s) {
        return s.topology.StorageLevelNames();
      });

  py::class_<model::Engine>(m, "Engine")
      .def(py::init<>(),
           "Construct wrapper over Timeloop's native Engine class. Consider "
           "using `pytimeloop.Accelerator` instead. \n"
           "Engine.spec has to be called later with ArchSpecs.")
      .def(py::init([](model::Engine::Specs specs) {
             auto e = std::make_unique<model::Engine>();
             e->Spec(specs);
             return e;
           }),
           "Construct and spec Engine.")
      .def("spec", &model::Engine::Spec)
      .def("pre_evaluation_check", &model::Engine::PreEvaluationCheck)
      .def("evaluate", &model::Engine::Evaluate, py::arg("mapping"),
           py::arg("workload"), py::arg("sparse_optimizations"),
           py::arg("break_on_failure") = true)
      .def("is_evaluated", &model::Engine::IsEvaluated)
      .def("utilization", &model::Engine::Utilization)
      .def("energy", &model::Engine::Energy)
      .def("area", &model::Engine::Area)
      .def("cycles", &model::Engine::Cycles)
      .def("get_topology", &model::Engine::GetTopology)
      .def("pretty_print_stats", [](model::Engine& e) -> std::string {
        std::stringstream ss;
        ss << e << std::endl;
        return ss.str();
      });
}

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

void BindLevel(py::module& m) {
  py::class_<model::EvalStatus>(m, "EvalStatus")
      .def_readonly("success", &model::EvalStatus::success)
      .def_readonly("fail_reason", &model::EvalStatus::fail_reason)
      .def("__repr__", [](model::EvalStatus& e) -> std::string {
        if (e.success) {
          return "EvalStatus(success=1)";
        } else {
          return "EvalStatus(success=0, fail_reason=" + e.fail_reason + ")";
        }
      });
}

void BindSparseOptimizationInfo(py::module& m) {
  py::class_<sparse::SparseOptimizationInfo>(m, "SparseOptimizationInfo")
      .def(py::init(&sparse::ParseAndConstruct))
      .def_static("parse_and_construct", &sparse::ParseAndConstruct,
                  py::call_guard<py::scoped_ostream_redirect,
                                 py::scoped_estream_redirect>());
}

void BindTopology(py::module& m) {
  py::class_<model::Topology>(m, "Topology")
      .def("algorithmic_computes", &model::Topology::AlgorithmicComputes)
      .def("actual_computes", &model::Topology::ActualComputes)
      .def("last_level_accesses", &model::Topology::LastLevelAccesses)
      .def("tile_sizes", &model::Topology::TileSizes)
      .def("utilized_capacities", &model::Topology::UtilizedCapacities);
}

}  // namespace pytimeloop::model_bindings
