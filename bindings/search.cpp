#include <optional>

#include "bindings.h"

// Timeloop headers
#include "search/search-factory.hpp"
#include "search/search.hpp"

void BindSearchClasses(py::module& m) {
  py::class_<search::SearchAlgorithm>(m, "NativeSearchAlgorithm")
      .def("next",
           [](search::SearchAlgorithm* s) -> std::optional<mapspace::ID> {
             mapspace::ID mapping_id;
             return s.Next(mapping_id) ? mapping_id : std::nullopt;
           })
      .def("report", &search::SearchAlgorithm::Report);
}
