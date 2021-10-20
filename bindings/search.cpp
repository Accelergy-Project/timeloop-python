#include "bindings/bindings.h"
#include "bindings/type_casters.h"

// Timeloop headers
#include "mapspaces/mapspace-base.hpp"
#include "search/hybrid.hpp"
#include "search/linear-pruned.hpp"
#include "search/random-pruned.hpp"
#include "search/search-factory.hpp"
#include "search/search.hpp"

class PySearchAlgorithm : public search::SearchAlgorithm {
 public:
  bool Next(mapspace::ID& mapping_id) override {
    PYBIND11_OVERRIDE_PURE(bool, SearchAlgorithm, Next, mapping_id);
  }

  void Report(search::Status status, double cost = 0) override {
    PYBIND11_OVERRIDE_PURE(void, SearchAlgorithm, Report, status, cost);
  }
};

void BindSearchClasses(py::module& m) {
  py::enum_<search::Status>(m, "SearchStatus")
      .value("Success", search::Status::Success)
      .value("MappingConstructionFailure",
             search::Status::MappingConstructionFailure)
      .value("EvalFailure", search::Status::EvalFailure);

  py::class_<search::SearchAlgorithm, PySearchAlgorithm>(
      m, "NativeSearchAlgorithm")
      .def_static("parse_and_construct", &search::ParseAndConstruct);

  py::class_<search::HybridSearch>(m, "HybridSearch")
      .def(
          py::init<config::CompoundConfigNode, mapspace::MapSpace*, unsigned>())
      .def("next",
           [](search::HybridSearch& s) -> std::optional<mapspace::ID> {
             mapspace::ID id;
             if (!s.Next(id)) return std::nullopt;
             return id;
           })
      .def("report", &search::HybridSearch::Report);

  py::class_<search::LinearPrunedSearch>(m, "LinearPrunedSearch")
      .def(
          py::init<config::CompoundConfigNode, mapspace::MapSpace*, unsigned>())
      .def("next",
           [](search::LinearPrunedSearch& s) -> std::optional<mapspace::ID> {
             mapspace::ID id;
             if (!s.Next(id)) return std::nullopt;
             return id;
           })
      .def("report", &search::LinearPrunedSearch::Report);

  py::class_<search::RandomPrunedSearch>(m, "RandomPrunedSearch")
      .def(
          py::init<config::CompoundConfigNode, mapspace::MapSpace*, unsigned>())
      .def("next",
           [](search::RandomPrunedSearch& s) -> std::optional<mapspace::ID> {
             mapspace::ID id;
             if (!s.Next(id)) return std::nullopt;
             return id;
           })
      .def("report", &search::RandomPrunedSearch::Report);
}
