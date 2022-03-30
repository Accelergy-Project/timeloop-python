#include "pytimeloop/bindings/mapper.h"

#include "pytimeloop/mapper/decoupled-mapper.h"

namespace pytimeloop::mapper_bindings {

void BindDecoupledMapper(py::module& m) {
  using namespace pytimeloop::pymapper;

  py::class_<DecoupledMapper>(m, "DecoupledMapper")
      .def(py::init<const model::Engine::Specs&, problem::Workload&,
                    std::vector<mapspace::MapSpace*>&,
                    std::vector<search::SearchAlgorithm*>&,
                    const sparse::SparseOptimizationInfo&,
                    const std::vector<std::string>&, unsigned, uint64_t,
                    unsigned, unsigned, bool>())
      .def("run", &DecoupledMapper::Run);
}

}  // namespace pytimeloop::mapper_bindings
