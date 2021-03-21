#include "bindings.h"

// Timeloop headers
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"
#include "mapping/mapping.hpp"
#include "mapping/parser.hpp"

void BindMappingClasses(py::module& m) {
  py::class_<ArchProperties>(m, "ArchProperties")
      .def(py::init<>())
      .def(py::init<const model::Engine::Specs&>());

  py::class_<mapping::Constraints>(m, "ArchConstraints")
      .def(py::init<const ArchProperties&, const problem::Workload&>())
      .def("parse", [](mapping::Constraints& c,
                       config::CompoundConfigNode config) { c.Parse(config); })
      .def("satisfied_by", &mapping::Constraints::SatisfiedBy);

  py::class_<Mapping>(m, "Mapping")
      .def_static("parse_and_construct", [](config::CompoundConfigNode mapping,
                                            model::Engine::Specs& archSpecs,
                                            problem::Workload& workload) {
        return mapping::ParseAndConstruct(mapping, archSpecs, workload);
      });
}
