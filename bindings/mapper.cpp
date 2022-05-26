#include "pytimeloop/bindings/mapper.h"

#include "pytimeloop/mapper/coupled-mapper.h"

namespace pytimeloop::mapper_bindings {

void BindDecoupledMapper(py::module& m) {
  using namespace pytimeloop::pymapper;

  py::class_<CoupledMapper>(m, "CoupledMapper")
      .def(py::init<
           const model::Engine::Specs&, problem::Workload&,
           std::vector<
               std::pair<mapspace::MapSpace*, search::SearchAlgorithm*>>&,
           sparse::SparseOptimizationInfo&, const std::vector<std::string>&,
           uint64_t, unsigned, unsigned, bool>())
      .def("run", &CoupledMapper::Run);
}

}  // namespace pytimeloop::mapper_bindings
