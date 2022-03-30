#include "pytimeloop/bindings/search.h"

#include "pytimeloop/bindings/type_casters.h"

// Timeloop headers
#include "mapspaces/mapspace-base.hpp"
#include "search/hybrid.hpp"
#include "search/linear-pruned.hpp"
#include "search/random-pruned.hpp"
#include "search/search-factory.hpp"
#include "search/search.hpp"

namespace pytimeloop::search_bindings {

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
  using namespace search;

  py::enum_<Status>(m, "SearchStatus")
      .value("Success", Status::Success)
      .value("MappingConstructionFailure", Status::MappingConstructionFailure)
      .value("EvalFailure", Status::EvalFailure);

  py::class_<SearchAlgorithm, PySearchAlgorithm>(m, "SearchAlgorithm")
      .def_static("parse_and_construct", &ParseAndConstruct);

  py::class_<HybridSearch, SearchAlgorithm>(m, "HybridSearch")
      .def(
          py::init<config::CompoundConfigNode, mapspace::MapSpace*, unsigned>())
      .def("next",
           [](HybridSearch& s) -> std::optional<mapspace::ID> {
             mapspace::ID id;
             if (!s.Next(id)) return std::nullopt;
             return id;
           })
      .def("report", &HybridSearch::Report);

  py::class_<LinearPrunedSearch, SearchAlgorithm>(m, "LinearPrunedSearch")
      .def(
          py::init<config::CompoundConfigNode, mapspace::MapSpace*, unsigned>())
      .def("next",
           [](LinearPrunedSearch& s) -> std::optional<mapspace::ID> {
             mapspace::ID id;
             if (!s.Next(id)) return std::nullopt;
             return id;
           })
      .def("report", &LinearPrunedSearch::Report);

  py::class_<RandomPrunedSearch, SearchAlgorithm>(m, "RandomPrunedSearch")
      .def(
          py::init<config::CompoundConfigNode, mapspace::MapSpace*, unsigned>())
      .def("next",
           [](RandomPrunedSearch& s) -> std::optional<mapspace::ID> {
             mapspace::ID id;
             if (!s.Next(id)) return std::nullopt;
             return id;
           })
      .def("report", &RandomPrunedSearch::Report);
}

}  // namespace pytimeloop::search_bindings
