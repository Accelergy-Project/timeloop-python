#include "pytimeloop/bindings/config.h"

#include <optional>
#include <variant>

// PyBind11 headers
#include "pybind11/stl.h"

// Timeloop headers
#include "compound-config/compound-config.hpp"
#include "pytimeloop/configurator/configurator.h"

namespace pytimeloop::config_bindings {

typedef std::optional<std::variant<bool, long long, unsigned long long, double, 
char, std::string, config::CompoundConfigNode>>
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
      /** @brief The default constructor. The only way you should get one with
       * values is through Config */
      .def(py::init<>())
      /// @brief Accession. Is used to traverse CCNs like a list or dict.
      .def("__getitem__", [](CompoundConfigNode& self, 
                             const std::variant<int, std::string>& keyIn) ->
                             CompoundConfigLookupReturn 
      {

        // Value resolution based in input type of key.
        const CompoundConfigNode& value = 
          std::holds_alternative<std::string>(keyIn) ?
            self.lookup(std::get<std::string>(keyIn)):
            self[std::get<int>(keyIn)];

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
            /* Attempts to resolve the scalar as a long long. Long long before
             * double to avoid FP issues where possible */
            try { return YNode.as<long long>(); } 
            catch(const YAML::TypedBadConversion<long long>& e) {}
            // Attempts to resolve the scalar as a double.
            try { return YNode.as<double>();}
            catch(const YAML::TypedBadConversion<double>& e) {}
            /** @todo A workaround for https://github.com/jbeder/yaml-cpp/issues/1198
             * Resolve via the following char emit for now, will open a PR in the
             * YAML-CPP repo to resolve in the future. There are no single char
             * things that map to values that are not strings, so this is only
             * a fix.
             */ 
            try { return YNode.as<char>(); }
            catch(const YAML::TypedBadConversion<char>& e) {}
            // Attempts to resolve the scalar as a bool.
            try { return YNode.as<bool>(); }
            catch(const YAML::TypedBadConversion<bool>& e) {}
            // Attempts to resolve the scalar as a string.
            try { return YNode.as<std::string>(); }
            catch(const YAML::TypedBadConversion<std::string>& e) {}
            // Error if we cannot resolve as any of the types above.
            throw std::runtime_error("Could not resolve this node to a scalar.");
            break;
          default:
            return value;
        }
      })
      /// @brief Setting. Is used to traverse CCNs like a list or dict.
      .def("__setitem__", [](CompoundConfigNode& self,
                             const std::variant<int, std::string>& keyIn,
                             CompoundConfigLookupReturn input)
      {
        // Instantiates the key if it does not exist and is currently a Map or Null
        if (self.isMap() || self.getYNode().IsNull())
        {
          self.instantiateKey(
            std::holds_alternative<std::string>(keyIn) ?
              std::get<std::string>(keyIn):
              std::to_string(std::get<int>(keyIn))
          );
        /* Does nothing if it is not a key because then we expect the index to
         * already be there */
        }

        // Current location resolution based in input type of key and current input.
        const CompoundConfigNode& loc =
          (self.isList() || self.isArray()) ?
            self[std::get<int>(keyIn)]:
            std::holds_alternative<std::string>(keyIn) ?
              self.lookup(std::get<std::string>(keyIn)):
              self.lookup(std::to_string(std::get<int>(keyIn)));
      })
      /// @brief Pushes an object onto a CompoundConfigNode if Null or Sequence.
      // .def("append", [](CompoundConfigNode& self, CompoundConfigLookupReturn val)
      // {
      //   self.push_back(val);
      // })

      /// @brief Converts the Node to a string.
      .def("__str__", [](CompoundConfigNode& self) {
        // Emitter is the intermediate layer to output strings.
        YAML::Emitter temp;
        // Loads Node into Emitter.
        temp << self.getYNode();

        return std::string(temp.c_str());
      })
      .def("resolve", &CompoundConfigNode::resolve);
} 

}  // namespace pytimeloop::config_bindings
