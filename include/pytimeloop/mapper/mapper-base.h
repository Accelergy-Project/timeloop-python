#pragma once

#include <utility>

// Timeloop library
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <search/search.hpp>

#include "pytimeloop/model/eval-result.h"
#include "pytimeloop/search/mapspace-search.h"

namespace pytimeloop::pymapper {

/**
 * Abstract base class of a Mapper.
 *
 */
class Mapper {
 public:
  virtual std::pair<Mapping, pytimeloop::pymodel::EvaluationResult> Run() = 0;

 protected:
  typedef model::Engine::Specs ArchSpecs;
  typedef problem::Workload Workload;
  typedef mapspace::MapSpace MapSpace;
  typedef search::SearchAlgorithm SearchAlgorithm;
  typedef sparse::SparseOptimizationInfo SparseOptInfo;
};

}  // namespace pytimeloop::pymapper
