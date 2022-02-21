#include "pytimeloop/bindings/mapper.h"

#include "pytimeloop/mapper/decoupled-mapper.h"

namespace mapper_bindings {

void BindDecoupledMapper(py::module& m) {
  py::class_<DecoupledMapper>(m, "DecoupledMapper")
      .def(py::init<const model::Engine::Specs&, problem::Workload&,
                    std::vector<mapspace::MapSpace*>&,
                    std::vector<search::SearchAlgorithm*>&,
                    const sparse::SparseOptimizationInfo&,
                    const std::vector<std::string>&, unsigned, uint64_t,
                    unsigned, unsigned, bool>())
      .def("run", &DecoupledMapper::Run);
}

}  // namespace mapper_bindings
