#pragma once

#include <string>
#include <variant>
#include <vector>

// Timeloop headers
#include "compound-config/compound-config.hpp"
#include "mapping/parser.hpp"
#include "model/level.hpp"

class PyCompoundConfigNode;

typedef std::variant<PyCompoundConfigNode, bool, long long, unsigned long long,
                     double, std::string, std::vector<std::string>>
    LookupReturn;

class PyCompoundConfigNode {
 public:
  PyCompoundConfigNode() {}
  PyCompoundConfigNode(config::CompoundConfigNode node) : node_(node) {}

  LookupReturn LookupValue(const std::string &key) const;
  LookupReturn operator[](int idx);
  bool Exists(const std::string &key) const;
  std::vector<std::string> GetMapKeys();

  const config::CompoundConfigNode GetUnderlying() const;

 private:
  config::CompoundConfigNode node_;
};
