#pragma once

#include "pytimeloop/model/eval-result.h"

namespace pytimeloop::pymodel {

enum Betterness { Better, SlightlyBetter, SlightlyWorse, Worse };

double Cost(const EvaluationResult& stats, const std::string metric);

bool IsBetter(const EvaluationResult& candidate,
              const EvaluationResult& incumbent,
              const std::vector<std::string>& metrics);

}  // namespace pytimeloop::pymodel
