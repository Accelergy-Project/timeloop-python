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
std::string, config::CompoundConfigNode>>
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
                             const std::variant<int, std::string>& keyIn)
      {

        // Current location resolution based in input type of key and current input.
        return (self.isList() || self.isArray()) ?
            self[std::get<int>(keyIn)]:
            std::holds_alternative<std::string>(keyIn) ?
              self.lookup(std::get<std::string>(keyIn)):
              self.lookup(std::to_string(std::get<int>(keyIn)));
      })
      /// @brief Setting. Is used to traverse CCNs like a list or dict.
      .def("__setitem__", [](CompoundConfigNode& self,
                             const std::variant<int, std::string>& keyIn,
                             CompoundConfigLookupReturn val)
      {
        // Instantiates the key if it does not exist and is currently a Map or Null
        if (self.isMap() || self.getYNode().IsNull())
        {
          // Instantiates key if it doesn't exist.
          self.instantiateKey(
            std::holds_alternative<std::string>(keyIn) ?
              std::get<std::string>(keyIn):
              std::to_string(std::get<int>(keyIn))
          );
        /* Does nothing if it is not a key because then we expect the index to
         * already be there */
        }

        // Current location resolution based in input type of key and current input.
        CompoundConfigNode loc =
          (self.isList() || self.isArray()) ?
            self[std::get<int>(keyIn)]:
            std::holds_alternative<std::string>(keyIn) ?
              self.lookup(std::get<std::string>(keyIn)):
              self.lookup(std::to_string(std::get<int>(keyIn)));

        // If val is Null, set nothing.
        if (val)
        {
          // Otherwise, unpack the value and assign.
          if (std::holds_alternative<std::string>(*val))
          {
            loc.setScalar(std::get<std::string>(*val));
          } else if (std::holds_alternative<double>(*val))
          {
            loc.setScalar(std::get<double>(*val));
          } else if (std::holds_alternative<long long>(*val))
          {
            loc.setScalar(std::get<long long>(*val));
          } else if (std::holds_alternative<bool>(*val))
          {
            loc.setScalar(std::get<bool>(*val));
          /* Assigns new YNodes like this because we pass YAML::Nodes by reference
            * Through novel CompoundConfigNodes, making editing loc bad since we
            * want to replace the child in the parent, not assign a value to the
            * child. */
          } else if (std::holds_alternative<CompoundConfigNode>(*val))
          {
            YAML::Node YNode = self.getYNode();
            YAML::Node child = std::get<CompoundConfigNode>(*val).getYNode();
            if (YNode.IsSequence())
            {
              YNode[std::get<int>(keyIn)] = child;
            } else
            {
              if (std::holds_alternative<int>(keyIn))
              {
                YNode[std::to_string(std::get<int>(keyIn))] = child;
              } else
              {
                YNode[std::get<std::string>(keyIn)] = child;
              }
            }
          } else {
            throw std::runtime_error("Tried to set YAML to an invalid type.");
          }
        } else
        {
          loc.setScalar(YAML::Null);
        }
      })
      /// @brief Pushes an object onto a CompoundConfigNode if Null or Sequence.
      .def("append", [](CompoundConfigNode& self, CompoundConfigLookupReturn val)
      {
        // If not Null resolve type.
        if (val)
        {
          if (std::holds_alternative<std::string>(*val))
          {
            self.push_back(std::get<std::string>(*val));
          } else if (std::holds_alternative<double>(*val))
          {
            self.push_back(std::get<double>(*val));
          } else if (std::holds_alternative<long long>(*val))
          {
            self.push_back(std::get<long long>(*val));
          } else if (std::holds_alternative<bool>(*val))
          {
            self.push_back(std::get<bool>(*val));
          } else if (std::holds_alternative<CompoundConfigNode>(*val)) 
          {
            YAML::Node child = std::get<CompoundConfigNode>(*val).getYNode();
            self.push_back(child);
          } else
          {
            throw std::runtime_error("Tried to append an inbalid YAML type.");
          }
        // If Null, pushback Null.
        } else
        {
          self.push_back(YAML::Node());
        }
      })

      /// @brief resolves a Node to a Scalar if possible.
      .def("resolve", [](CompoundConfigNode& self) -> CompoundConfigLookupReturn 
      {
        // Extracts the YNode inside it.
        const YAML::Node& YNode = self.getYNode();

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
            try { return std::string(1, YNode.as<char>()); }
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
            return self;
        }
      })

      /// @brief Converts the Node to a string.
      .def("__str__", [](CompoundConfigNode& self) {
        // Emitter is the intermediate layer to output strings.
        YAML::Emitter temp;
        // Loads Node into Emitter.
        temp << self.getYNode();

        return std::string(temp.c_str());
      });
} 

}  // namespace pytimeloop::config_bindings
