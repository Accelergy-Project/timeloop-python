#include "bindings.h"

// Timeloop headers
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"
#include "mapping/mapping.hpp"
#include "mapping/parser.hpp"

// Type casters
#include "type_casters.h"

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
      .def_static(
          "parse_and_construct",
          [](config::CompoundConfigNode mapping,
             model::Engine::Specs& archSpecs, problem::Workload& workload) {
            return mapping::ParseAndConstruct(mapping, archSpecs, workload);
          })
      .def(
          "pretty_print",
          [](Mapping& m, py::object py_out,
             const std::vector<std::string>& storage_level_names,
             const std::vector<problem::PerDataSpace<std::uint64_t>>&
                 tile_sizes,
             const std::string _indent) {
            std::ostringstream out;
            py::scoped_ostream_redirect redirect(out, py_out);
            m.PrettyPrint(out, storage_level_names, tile_sizes, _indent);
          },
          py::arg(), py::arg(), py::arg(), py::arg("indent") = "");
}
