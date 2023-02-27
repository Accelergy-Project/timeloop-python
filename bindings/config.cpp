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

#ifdef NEW_CONFIG_INTERFACE
  using CompoundConfig = config::CompoundConfig;
  py::class_<CompoundConfig>(m, "Config")
      .def(py::init([]() { return CompoundConfig(); }))
      .def_static("from_yaml_str", [](const std::string& yaml_str) {
        return CompoundConfig(yaml_str, "yaml");
      })
      .def("new_dict", &CompoundConfig::NewDict,
           py::return_value_policy::reference_internal)
      .def("new_list", &CompoundConfig::NewList,
           py::return_value_policy::reference_internal)
      .def("new_primitive", &CompoundConfig::NewPrimitive<std::string>,
           py::return_value_policy::reference_internal)
      .def("new_primitive", &CompoundConfig::NewPrimitive<int>,
           py::return_value_policy::reference_internal)
      .def("new_primitive", &CompoundConfig::NewPrimitive<double>,
           py::return_value_policy::reference_internal)
      .def("new_primitive", &CompoundConfig::NewPrimitive<bool>,
           py::return_value_policy::reference_internal)
      .def("insert", &CompoundConfig::Insert)
      .def("append", &CompoundConfig::Append)
      .def("lookup", &CompoundConfig::Lookup)
      .def("lookup", &CompoundConfig::Lookup<std::string>)
      .def("lookup", &CompoundConfig::Lookup<int>)
      .def("lookup", &CompoundConfig::Lookup<double>)
      .def("lookup", &CompoundConfig::Lookup<bool>)
      .def("exists", &CompoundConfig::Exists)
      .def("size", &CompoundConfig::Size);

  using CompoundConfigNode = config::CompoundConfigNode;
  py::class_<CompoundConfigNode>(m, "ConfigNode");
#endif
}

}  // namespace pytimeloop::config_bindings
