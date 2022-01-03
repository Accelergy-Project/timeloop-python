#pragma once

#include <memory>
#include <vector>

// Timeloop library
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <search/search.hpp>

#include "bindings/search/mapspace-search.h"

/**
 * Abstract base class of a Mapper.
 *
 */
class Mapper {
 public:
  virtual Mapping Run() = 0;

 protected:
  typedef model::Engine::Specs ArchSpecs;
  typedef problem::Workload Workload;
  typedef mapspace::MapSpace MapSpace;
  typedef search::SearchAlgorithm SearchAlgorithm;
  typedef sparse::SparseOptimizationInfo SparseOptInfo;
};
