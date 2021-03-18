#include <variant>
#include <vector>

// Timeloop headers
#include "compound-config/compound-config.hpp"
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"
#include "mapping/mapping.hpp"
#include "model/engine.hpp"
#include "util/accelergy_interface.hpp"
#include "workload/problem-shape.hpp"
#include "workload/workload.hpp"

// PyBind11 headers
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

// Python wrapper classes
#include "config.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

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
      .def("get_root", [](config::CompoundConfig& c) {
        return PyCompoundConfigNode(c.getRoot());
      });

  py::class_<PyCompoundConfigNode>(m, "ConfigNode")
      .def(py::init<>())
      .def("__getitem__", &PyCompoundConfigNode::LookupValue)
      .def("__getitem__", &PyCompoundConfigNode::operator[])
      .def("get", &PyCompoundConfigNode::LookupValue)
      .def("__contains__", &PyCompoundConfigNode::Exists)
      .def("keys", &PyCompoundConfigNode::GetMapKeys);

  py::class_<problem::Workload>(m, "Workload")
      .def(py::init<>())
      .def("parse_workload",
           [](problem::Workload& w, PyCompoundConfigNode& config) {
             problem::ParseWorkload(config.GetUnderlying(), w);
           });

  py::class_<model::Engine::Specs>(m, "ArchSpecs")
      .def_static(
          "parse_specs",
          [](PyCompoundConfigNode& specConfig) {
            return model::Engine::ParseSpecs(specConfig.GetUnderlying());
          })
      .def("parse_accelergy_art",
           [](model::Engine::Specs& s, PyCompoundConfigNode& art) {
             s.topology.ParseAccelergyART(art.GetUnderlying());
           })
      .def("parse_accelergy_ert",
           [](model::Engine::Specs& s, PyCompoundConfigNode& ert) {
             s.topology.ParseAccelergyERT(ert.GetUnderlying());
           })
      .def("level_names",
           [](model::Engine::Specs& s) { return s.topology.LevelNames(); });

  py::class_<ArchProperties>(m, "ArchProperties")
      .def(py::init<>())
      .def(py::init<const model::Engine::Specs&>());

  py::class_<mapping::Constraints>(m, "ArchConstraints")
      .def(py::init<const ArchProperties&, const problem::Workload&>())
      .def("parse",
           [](mapping::Constraints& c, PyCompoundConfigNode config) {
             c.Parse(config.GetUnderlying());
           })
      .def("satisfied_by", &mapping::Constraints::SatisfiedBy);

  py::class_<Mapping>(m, "Mapping")
      .def_static("parse_and_construct", [](PyCompoundConfigNode mapping,
                                            model::Engine::Specs& archSpecs,
                                            problem::Workload& workload) {
        return Mapping(mapping::ParseAndConstruct(mapping.GetUnderlying(),
                                                  archSpecs, workload));
      });

  py::class_<model::Engine>(m, "Engine")
      .def("spec", &model::Engine::Spec)
      .def("pre_evaluation_check", &model::Engine::PreEvaluationCheck)
      .def("evaluate", &model::Engine::Evaluate);

  py::class_<problem::Shape>(m, "ProblemShape");

  m.def("get_problem_shape", &problem::GetShape);

  m.def("invoke_accelergy", &accelergy::invokeAccelergy);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
