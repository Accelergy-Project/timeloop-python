#include "pytimeloop/bindings/config.h"

#include <optional>
#include <variant>

// PyBind11 headers
#include "pybind11/stl.h"

// Timeloop headers
#include "compound-config/compound-config.hpp"
#include "pytimeloop/configurator/configurator.h"

namespace pytimeloop::config_bindings {

typedef std::optional<std::variant<bool, long long, unsigned long long, double, std::string,
config::CompoundConfigNode>>
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

  /// @brief Creates an equivalent CompoundConfig class in Python.
  using CompoundConfig = config::CompoundConfig;
  py::class_<CompoundConfig>(m, "Config")
      /// @brief Initializer. Uses the CompoundConfig string + type constructor.
      .def(py::init<std::string &, std::string &>())
      /// @brief Fetches the root CompoundConfigNode.
      .def("getRoot", &CompoundConfig::getRoot);
  
  /// @brief Creates an equivalent CompoundConfigNode class in Python. 
  using CompoundConfigNode = config::CompoundConfigNode;
  py::class_<CompoundConfigNode>(m, "ConfigNode")
      /// @brief Accession. Is used to traverse CCNs like a list or dict.
      .def("__getitem__", [](CompoundConfigNode& self, 
                             const std::variant<int, std::string> keyIn) ->
                             CompoundConfigLookupReturn 
      {
        // Execution path if the key is a string (CCN is a Map).
        if (std::holds_alternative<std::string>(keyIn))
        {
          // Collapses value to string.
          std::string key = std::get<std::string>(keyIn);
          // Extracts value node from self.
          const CompoundConfigNode& value = self.lookup(key);
          // Extracts the YNode inside it.
          const YAML::Node& YNode = value.getYNode();

          // If the value node is not a Scalar, return itself.
          switch (YNode.Type())
          {
            // If you're a Null node, you're the equivalent of a None.
            case YAML::NodeType::Null:
              return std::nullopt;
              break;
            // Attempts to resolve the scalar to one of Python's types.
            case YAML::NodeType::Scalar:
              // Attempts to resolve the scalar as a bool.
              try { return YNode.as<bool>(); }
              catch(const YAML::TypedBadConversion<bool>& e) {}
              // Attempts to resolve the scalar as a double.
              try { return YNode.as<double>();}
              catch(const YAML::TypedBadConversion<double>& e) {}
              // Attempts to resolve the scalar as a long long.
              try { return YNode.as<long long>(); } 
              catch(const YAML::TypedBadConversion<long long>& e) {}
              // Attempts to resolve the scalar as a string.
              try { return YNode.as<std::string>(); }
              catch(const YAML::TypedBadConversion<std::string>& e) {}
              // Error if we cannot resolve as any of the types above.
              throw std::runtime_error("Could not resolve this node to a scalar.");
              break;
            default:
              return value;
          }
        // Execution path if CCN is a int (CCN is a list)
        } else
        {
          return self[std::get<int>(keyIn)];
        }

      })
      .def("resolve", &CompoundConfigNode::resolve);
} 

}  // namespace pytimeloop::config_bindings
