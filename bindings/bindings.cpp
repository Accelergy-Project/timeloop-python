#include <variant>
#include <vector>

// PyBind11 headers
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

// Python wrapper classes
#include "config.h"
#include "workload.h"

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

  py::class_<PyCompoundConfig>(m, "Config")
      .def(py::init<std::string &>())
      .def(py::init<std::vector<std::string>>())
      .def("get_root", &PyCompoundConfig::GetRoot);

  py::class_<PyCompoundConfigNode>(m, "ConfigNode")
      .def("__getitem__", &PyCompoundConfigNode::LookupValue)
      .def("__getitem__", &PyCompoundConfigNode::operator[])
      .def("get", &PyCompoundConfigNode::LookupValue)
      .def("__contains__", &PyCompoundConfigNode::Exists)
      .def("keys", &PyCompoundConfigNode::GetMapKeys);

  py::class_<PyWorkload>(m, "Workload")
      .def(py::init<PyCompoundConfigNode>())
      .def("coefficient", &PyWorkload::GetCoefficient)
      .def("density", &PyWorkload::GetDensity);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
