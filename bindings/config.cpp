// std's necessary for pybind11.
#include <optional>
#include <variant>

// Timeloop headers
#include "compound-config/compound-config.hpp"

// PyTimeloop headers
#include "pytimeloop/bindings/config.h"
#include "pytimeloop/configurator/configurator.h"

namespace pytimeloop::config_bindings {

using CompoundConfigLookupReturn = std::optional<std::variant<
  bool, long long, unsigned long long, double, std::string, config::CompoundConfigNode
>>;

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
  py::class_<CompoundConfig> config(m, "Config");

  config.doc() = R"DOCSTRING(
    @brief  The configuration object Timeloop uses to determine how to run. 
    
    This is wrapping a ConfigNode for ownership reasons. ConfigNode is a 
    YAML::Node with some extra functionality to keep backwards compatibility 
    with .cfg files. PyTimeloop, however, only supports YAML files.
  )DOCSTRING";

  config
      /// @brief Initializer. Uses the CompoundConfig string + type constructor.
      .def(py::init<std::string &, std::string &>())
      /// @brief Fetches the root CompoundConfigNode.
      .def_property_readonly("root", &CompoundConfig::getRoot);
  
  /// @brief Creates an equivalent Config.CompoundConfigNode class in Python. 
  using CompoundConfigNode = config::CompoundConfigNode;
  py::class_<CompoundConfigNode> config_node(config, "ConfigNode");

  config_node.doc() = R"DOCSTRING(
    @brief  The configuration node Timeloop uses to determine how to run.

    This is a structure that wraps a YAML::Node, in order to keep backwards
    compatibility in Timeloop with .cfg files. PyTimeloop, however, only
    supports YAML files. This class is also used to traverse the YAML tree
    in a way that is more Pythonic.
  )DOCSTRING";

  config_node
      /** @brief The default constructor. The only way you should get one with
       * values is through Config */
      .def(py::init<>())
      /// @brief Accession. Is used to traverse CCNs like a list or dict.
      .def("__getitem__", [](CompoundConfigNode& self, 
                             const std::variant<int, std::string>& keyIn) ->
                             CompoundConfigNode
      {
        // Current location resolution based in input type of key and current input.
        return (self.isList() || self.isArray()) ?
            self[std::get<int>(keyIn)]:
            std::holds_alternative<std::string>(keyIn) ?
              self.lookup(std::get<std::string>(keyIn)):
              self.lookup(std::to_string(std::get<int>(keyIn)));
      },
      R"DOCSTRING(
        Returns the value at the given key. Can accept both strings or ints, and
        will resolve to different behavior depending on what type of ConfigNode 
        we are currently in.
        
        @param self   The current ConfigNode.
        @param keyIn  The key/index we want to try to access.

        @return       The ConfigNode at the given key/index.
      )DOCSTRING")
      /// @brief Setting. Is used to traverse CCNs like a list or dict.
      .def("__setitem__", [](CompoundConfigNode& self,
                             const std::variant<int, std::string>& keyIn,
                             CompoundConfigLookupReturn val) -> void
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

        // Value location resolution based on input type of key and current location.
        CompoundConfigNode loc =
          (self.isList() || self.isArray()) ?
            self[std::get<int>(keyIn)]:
            std::holds_alternative<std::string>(keyIn) ?
              self.lookup(std::get<std::string>(keyIn)):
              self.lookup(std::to_string(std::get<int>(keyIn)));

        // If val is Null, set nothing.
        if (val)
        {
          /* Otherwise, unpack the value and assign. */
          // String assignment.
          if (std::holds_alternative<std::string>(*val))
          {
            loc.setScalar(std::get<std::string>(*val));
          // FP assignment.
          } else if (std::holds_alternative<double>(*val))
          {
            loc.setScalar(std::get<double>(*val));
          // Int assignment.
          } else if (std::holds_alternative<long long>(*val))
          {
            loc.setScalar(std::get<long long>(*val));
          // Bool assignment.
          } else if (std::holds_alternative<bool>(*val))
          {
            loc.setScalar(std::get<bool>(*val));
          /* YNode assignment. Assigns new YNodes like this because we pass 
           * YAML::Nodes by reference through novel CompoundConfigNodes, making 
           * editing loc bad since we want to replace the child in the parent, 
           * not assign a value to the child which will then get immediately
           * destructed without mutating the struct we want to mutate. */
          } else if (std::holds_alternative<CompoundConfigNode>(*val))
          {
            // Unpacks the current YNode in CompoundConfigNode.
            YAML::Node YNode = self.getYNode();
            // Unpacks the child YAML::Node we want to assign to YNode.
            YAML::Node child = std::get<CompoundConfigNode>(*val).getYNode();

            // If we are currently a Sequence, assign like a Sequence.
            if (YNode.IsSequence())
            {
              YNode[std::get<int>(keyIn)] = child;
            // Otherwise, assign like a Map.
            } else
            {
              // We want all Map keys to be strings, so make the key a string.
              YNode[std::holds_alternative<int>(keyIn)?
                    std::to_string(std::get<int>(keyIn)):
                    std::get<std::string>(keyIn)] = child;
            }
          /* Throws an error if we have a type not handled above. That means
           * the Python invocation has some deep error. */
          } else 
          {
            throw std::runtime_error("Tried to set YAML to an invalid type.");
          }
        // If val is Null, set it to Null.
        } else
        {
          loc.setScalar(YAML::Null);
        }
      },
      R"DOCSTRING(
        Sets the value at the given key. Can accept both strings or ints as keys.
        Values can be any return type from __getitem__ (scalars or a ConfigNode).

        Like with Python dicts, if the key does not exist, it will be created. If
        the ConfigNode is instead a list, if a index is not present, it will throw
        an error.

        @param self   The current ConfigNode.
        @param keyIn  The key we would like to set.
      )DOCSTRING")
      /// @brief Makes it so the in command in Python works; only for Maps.
      .def("__contains__", [](CompoundConfigNode& self, const std::string& key)
      -> bool
      {
        // Only returns true if self is a Map and the val we're matching is a key.
        return self.isMap() && self.exists(key);
      }, 
      "Returns true if the given key exists in the current ConfigNode.")
      /// @brief Pushes an object onto a CompoundConfigNode if Null or Sequence.
      .def("append", [](CompoundConfigNode& self, CompoundConfigLookupReturn val)
      -> void
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
      },
      R"DOCSTRING(
        Appends the given value to the end of the current ConfigNode if it's a
        list. If the current ConfigNode is Null, it will be converted to a list
        and the value we are trying to append will be at index 0.
      
        @param self  The current ConfigNode.
        @param val   The value we want to append.
      )DOCSTRING")
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
      },
      R"DOCSTRING(
        Attempts to resolve self to a scalar when possible.

        @param self   The current ConfigNode.
        @return       The resolved scalar or the current ConfigNode if it cannot
                      be resolved.
      )DOCSTRING")
      /// @brief Converts the Node to a string.
      .def("__str__", [](CompoundConfigNode& self) -> std::string
      {
        // Emitter is the intermediate layer to output strings.
        YAML::Emitter temp;
        // Loads Node into Emitter.
        temp << self.getYNode();

        return std::string(temp.c_str());
      },
      "Emits the current ConfigNode as a string.");
} 
}  // namespace pytimeloop::config_bindings
