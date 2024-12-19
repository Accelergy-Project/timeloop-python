#include "pytimeloop/bindings/visualization.h"

#include <string>

#include <pybind11/stl.h>

#include <isl-wrapper/ctx-manager.hpp>

#include "pytimeloop/visualization/occupancy.h"


namespace pytimeloop::visualization_bindings
{
  void BindVisualization(py::module& m)
  {
    py::class_<OccupancyMatrix>(m, "OccupancyMatrix")
      .def_readonly("max_occupancy", &OccupancyMatrix::max_occupancy)
      .def_readonly("max_temporal_steps", &OccupancyMatrix::max_temporal_steps)
      .def_readonly("data", &OccupancyMatrix::data);

    m.def(
      "compute_occupancy_matrix",
      [](const std::map<int, std::string>& occupancy)
      {
        std::map<int, isl_pw_qpolynomial*> isl_occupancy;
        isl_ctx* p_ctx = GetIslCtx().get();
        for (auto [tensor_id, qpolynomial_str] : occupancy)
        {
          isl_occupancy[tensor_id] = isl_pw_qpolynomial_read_from_str(
            p_ctx,
            qpolynomial_str.c_str()
          );
        }

        return ComputeOccupancyMatrix(isl_occupancy);
      }
    )
  }
}