#include <variant>
#include <vector>

#define USE_ACCELERGY

// Timeloop headers
#include "compound-config/compound-config.hpp"
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"
#include "mapping/mapping.hpp"
#include "model/engine.hpp"
#include "model/level.hpp"
#include "util/accelergy_interface.hpp"
#include "workload/problem-shape.hpp"
#include "workload/workload.hpp"

// PyBind11 headers
#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

// Python wrapper classes
#include "config.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

typedef std::variant<bool, long long, unsigned long long, double, std::string,
                     config::CompoundConfigNode>
    CompoundConfigLookupReturn;

PYBIND11_MODULE(bindings, m) {
  m.doc() = R"pbdoc(
        PyTimeloop bindings to C++ timeloop code
        -----------------------
        .. currentmodule:: pytimeloop
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

  py::class_<config::CompoundConfig>(m, "Config")
      .def(py::init<char*>())
      .def(py::init<std::vector<std::string>>())
      .def_readonly("in_files", &config::CompoundConfig::inFiles)
      .def("get_root", &config::CompoundConfig::getRoot)
      .def("get_variable_root", &config::CompoundConfig::getVariableRoot);

  py::class_<config::CompoundConfigNode>(m, "ConfigNode")
      .def(py::init<>())
      .def("__getitem__",
           [](config::CompoundConfigNode& n,
              std::string key) -> CompoundConfigLookupReturn {
             if (!n.exists(key)) {
               throw py::key_error(key);
             }

             bool bool_res;
             if (n.lookupValue(key, bool_res)) return bool_res;
             long long int_res;
             if (n.lookupValue(key, int_res)) return int_res;
             unsigned long long uint_res;
             if (n.lookupValue(key, uint_res)) return uint_res;
             double float_res;
             if (n.lookupValue(key, float_res)) return float_res;
             std::string string_res;
             if (n.lookupValue(key, string_res)) return string_res;

             return n.lookup(key);
           })
      .def("__getitem__",
           [](config::CompoundConfigNode& n,
              int idx) -> CompoundConfigLookupReturn {
             if (n.isArray()) {
               std::vector<std::string> resultVector;
               if (n.getArrayValue(resultVector)) {
                 return resultVector[idx];
               }
               throw py::index_error("Failed to access array.");
             }
             if (n.isList()) {
               return n[idx];
             }
             throw py::index_error("Not a list nor an array.");
           })
      .def("lookup", [](config::CompoundConfigNode& n,
                        std::string key) { return n.lookup(key); })
      .def("__contains__", [](config::CompoundConfigNode& n,
                              std::string key) { return n.exists(key); })
      .def("keys", [](config::CompoundConfigNode& n) {
        std::vector<std::string> all_keys;
        n.getMapKeys(all_keys);
        return all_keys;
      });

  // py::class_<PyCompoundConfigNode>(m, "ConfigNode")
  //     .def(py::init<>())
  //     .def("__getitem__", &PyCompoundConfigNode::LookupValue)
  //     .def("__getitem__", &PyCompoundConfigNode::operator[])
  //     .def("get", &PyCompoundConfigNode::LookupValue)
  //     .def("__contains__", &PyCompoundConfigNode::Exists)
  //     .def("keys", &PyCompoundConfigNode::GetMapKeys);

  py::class_<problem::Workload>(m, "Workload")
      .def(py::init<>())
      .def("parse_workload",
           [](problem::Workload& w, config::CompoundConfigNode& config) {
             problem::ParseWorkload(config, w);
           });

  py::class_<model::Engine::Specs>(m, "ArchSpecs")
      .def_static("parse_specs",
                  [](config::CompoundConfigNode& specConfig) {
                    return model::Engine::ParseSpecs(specConfig);
                  })
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
           [](model::Engine::Specs& s) { return s.topology.LevelNames(); });

  py::class_<ArchProperties>(m, "ArchProperties")
      .def(py::init<>())
      .def(py::init<const model::Engine::Specs&>());

  py::class_<mapping::Constraints>(m, "ArchConstraints")
      .def(py::init<const ArchProperties&, const problem::Workload&>())
      .def("parse", [](mapping::Constraints& c,
                       config::CompoundConfigNode config) { c.Parse(config); })
      .def("satisfied_by", &mapping::Constraints::SatisfiedBy);

  py::class_<Mapping>(m, "Mapping")
      .def_static("parse_and_construct", [](config::CompoundConfigNode mapping,
                                            model::Engine::Specs& archSpecs,
                                            problem::Workload& workload) {
        return Mapping(
            mapping::ParseAndConstruct(mapping, archSpecs, workload));
      });

  py::class_<model::Engine>(m, "Engine")
      .def(py::init<>())
      .def("spec", &model::Engine::Spec)
      .def("pre_evaluation_check", &model::Engine::PreEvaluationCheck)
      .def("evaluate", &model::Engine::Evaluate, py::arg("mapping"),
           py::arg("workload"), py::arg("break_on_failure") = true)
      .def("is_evaluated", &model::Engine::IsEvaluated)
      .def("utilization", &model::Engine::Utilization)
      .def("energy", &model::Engine::Energy)
      .def("get_topology", &model::Engine::GetTopology);

  py::class_<model::Topology>(m, "Topology")
      .def("maccs", &model::Topology::MACCs);

  py::class_<problem::Shape>(m, "ProblemShape")
      .def_readonly("num_data_spaces", &problem::Shape::NumDataSpaces);

  m.def("get_problem_shape", &problem::GetShape);

  m.def("invoke_accelergy", &accelergy::invokeAccelergy,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>());

  py::class_<model::EvalStatus>(m, "EvalStatus")
      .def_readonly("success", &model::EvalStatus::success)
      .def_readonly("fail_reason", &model::EvalStatus::fail_reason);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
