#include "config.h"

#include <optional>

// PyBind11 headers
#include "pybind11/pybind11.h"

// Timeloop headers
#include "compound-config/compound-config.hpp"

namespace py = pybind11;

LookupReturn PyCompoundConfigNode::LookupValue(const std::string &key) const {
  if (!node_.exists(key)) {
    throw py::key_error(key);
  }

  bool bool_res;
  if (node_.lookupValue(key, bool_res)) return bool_res;
  long long int_res;
  if (node_.lookupValue(key, int_res)) return int_res;
  unsigned long long uint_res;
  if (node_.lookupValue(key, uint_res)) return uint_res;
  double float_res;
  if (node_.lookupValue(key, float_res)) return float_res;
  std::string string_res;
  if (node_.lookupValue(key, string_res)) return string_res;

  return PyCompoundConfigNode(node_.lookup(key));
}

LookupReturn PyCompoundConfigNode::operator[](int idx) {
  if (node_.isArray()) {
    std::vector<std::string> resultVector;
    if (node_.getArrayValue(resultVector)) {
      return resultVector[idx];
    }
    throw py::index_error("Failed to access array.");
  }
  if (node_.isList()) {
    return PyCompoundConfigNode(node_[idx]);
  }
  throw py::index_error("Not a list nor an array.");
}

bool PyCompoundConfigNode::Exists(const std::string &key) const {
  return node_.exists(key);
}

std::vector<std::string> PyCompoundConfigNode::GetMapKeys() {
  std::vector<std::string> all_keys;
  node_.getMapKeys(all_keys);
  return all_keys;
}

config::CompoundConfigNode PyCompoundConfigNode::GetUnderlying() const {
  return node_;
}
