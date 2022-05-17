#include "pytimeloop/model/util.h"

namespace pytimeloop::pymodel {

Betterness IsBetterRecursive_(
    const EvaluationResult& candidate, const EvaluationResult& incumbent,
    const std::vector<std::string>::const_iterator metric,
    const std::vector<std::string>::const_iterator end);

double Cost(const EvaluationResult& stats, const std::string metric) {
  if (metric == "delay")
    return stats.cycles;
  else if (metric == "energy")
    return stats.energy;
  else if (metric == "last-level-accesses")
    return stats.last_level_accesses;
  else {
    assert(metric == "edp");
    return stats.energy * stats.cycles;
  }
}

bool IsBetter(const EvaluationResult& candidate,
              const EvaluationResult& incumbent,
              const std::vector<std::string>& metrics) {
  auto b =
      IsBetterRecursive_(candidate, incumbent, metrics.begin(), metrics.end());
  return (b == Betterness::Better || b == Betterness::SlightlyBetter);
}

Betterness IsBetterRecursive_(
    const EvaluationResult& candidate, const EvaluationResult& incumbent,
    const std::vector<std::string>::const_iterator metric,
    const std::vector<std::string>::const_iterator end) {
  const double tolerance = 0.001;

  double candidate_cost = Cost(candidate, *metric);
  double incumbent_cost = Cost(incumbent, *metric);

  double relative_improvement =
      incumbent_cost == 0 ? 1.0
                          : (incumbent_cost - candidate_cost) / incumbent_cost;

  if (fabs(relative_improvement) > tolerance) {
    // We have a clear winner.
    if (relative_improvement > 0)
      return Betterness::Better;
    else
      return Betterness::Worse;
  } else {
    // Within tolerance range, try to recurse.
    if (std::next(metric) == end) {
      // Base case. NOTE! Equality is categorized as SlightlyWorse (prefers
      // incumbent).
      if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    } else {
      // Recursive call.
      Betterness lsm =
          IsBetterRecursive_(candidate, incumbent, std::next(metric), end);
      if (lsm == Betterness::Better || lsm == Betterness::Worse) return lsm;
      // NOTE! Equality is categorized as SlightlyWorse (prefers incumbent).
      else if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    }
  }
}

}  // namespace pytimeloop::pymodel
