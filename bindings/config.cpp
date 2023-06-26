#include "pytimeloop/bindings/config.h"

#include <optional>
#include <variant>

// PyBind11 headers
#include "pybind11/stl.h"

// Timeloop headers
#include "compound-config/compound-config.hpp"
#include "pytimeloop/configurator/configurator.h"

namespace pytimeloop::config_bindings {

typedef std::variant<bool, long long, unsigned long long, double, std::string>
    CompoundConfigLookupReturn;

void BindConfigClasses(py::module& m) {
  using Configurator = pytimeloop::configurator::Configurator;
  py::class_<Configurator>(m, "Configurator")
      .def_static("from_yaml_str", &Configurator::FromYamlStr)
      .def("get_arch_props", &Configurator::GetArchProperties)
      .def("get_arch_specs", &Configurator::GetArchSpecs)
      .def("get_mapping", &Configurator::GetMapping)
      .def("get_mapping_constraints", &Configurator::GetMappingConstraints)
      .def("get_sparse_opts", &Configurator::GetSparseOptimizations)
      .def("get_workload",
           &Configurator::GetWorkload,
           py::return_value_policy::reference_internal);

  using CompoundConfig = config::CompoundConfig;
  py::class_<CompoundConfig>(m, "Config")
      .def(py::init<std::string &, std::string &>());
  
  /// @brief Creates an equivalent CompoundConfigNode class in Python. 
  using CompoundConfigNode = config::CompoundConfigNode;
  py::class_<CompoundConfigNode>(m, "ConfigNode")
      .def_static("__item__", [](CompoundConfigNode & self, const std::string& key) {
        return self.lookup(key.c_str());
      })
      .def("resolve", &CompoundConfigNode::resolve);
} 

}  // namespace pytimeloop::config_bindings
