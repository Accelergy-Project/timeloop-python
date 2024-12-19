#pragma once

#include <map>
#include <vector>

#include <isl/polynomial.h>


/**
 * The occupancy matrix contains tensor id as elements. The `occupancy_matrix`
 * member is in column-major format, starting from the * bottom left corner.
 * Each row has `max_temporal_steps` elements.
 */
struct OccupancyMatrix
{
  long max_occupancy;
  long max_temporal_steps;
  std::vector<std::vector<int>> data;
};


OccupancyMatrix ComputeOccupancyMatrix(
  const std::map<int, isl_pw_qpolynomial*>& occupancy
);
