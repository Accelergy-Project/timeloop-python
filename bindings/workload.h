#pragma once

#include <string>
#include <variant>
#include <vector>

// Timeloop headers
#include "loop-analysis/point-set.hpp"
#include "workload/problem-shape.hpp"
#include "workload/workload.hpp"

// Python wrapper classes
#include "config.h"

class PyWorkload {
 public:
  PyWorkload(PyCompoundConfigNode configNode) {
    problem::ParseWorkload(configNode.GetUnderlying(), workload_);
  };

  const problem::Shape *GetShape() const { return workload_.GetShape(); }
  int GetBound(problem::Shape::DimensionID dim) const {
    return workload_.GetBound(dim);
  }
  int GetCoefficient(problem::Shape::CoefficientID p) const {
    return workload_.GetCoefficient(p);
  }
  problem::DataDensity GetDensity(problem::Shape::DataSpaceID pv) const {
    return workload_.GetDensity(pv);
  }
  void SetBounds(const problem::Workload::Bounds &bounds) {
    return workload_.SetBounds(bounds);
  }
  void SetCoefficients(const problem::Workload::Coefficients &coef) {
    return workload_.SetCoefficients(coef);
  }
  void SetDensities(const problem::Workload::Densities &densities) {
    return workload_.SetDensities(densities);
  }

  const problem::Workload &GetUnderlying() const;

 private:
  problem::Workload workload_;
};
