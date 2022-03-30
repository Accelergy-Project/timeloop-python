#include "pytimeloop/bindings/mapspace.h"

#include "pytimeloop/bindings/type_casters.h"
#include "pytimeloop/mapspace/status.h"

// Timeloop headers
#include "mapspaces/mapspace-base.hpp"
#include "mapspaces/mapspace-factory.hpp"
#include "mapspaces/uber.hpp"
#include "workload/workload.hpp"

namespace pytimeloop::mapspace_bindings {

class PyMapSpace : public mapspace::MapSpace {
 public:
  using mapspace::MapSpace::MapSpace;
  std::vector<mapspace::Status> ConstructMapping(
      mapspace::ID mapping_id, Mapping* mapping,
      bool break_on_failure = true) override {
    PYBIND11_OVERRIDE_PURE(std::vector<mapspace::Status>, MapSpace,
                           ConstructMapping, mapping_id, mapping,
                           break_on_failure);
  }

  void InitPruned(uint128_t local_index_factorization_id) override {
    PYBIND11_OVERRIDE_PURE(void, MapSpace, InitPruned,
                           local_index_factorization_id);
  }

  std::vector<MapSpace*> Split(std::uint64_t num_splits) override {
    PYBIND11_OVERRIDE_PURE(std::vector<MapSpace*>, MapSpace, Split, num_splits);
  }
};

void BindMapspaceClasses(py::module& m) {
  using namespace pytimeloop::pymapspace;

  py::enum_<mapspace::Dimension>(m, "Dimension")
      .value("IndexFactorization", mapspace::Dimension::IndexFactorization)
      .value("LoopPermutation", mapspace::Dimension::LoopPermutation)
      .value("Spatial", mapspace::Dimension::Spatial)
      .value("DatatypeBypass", mapspace::Dimension::DatatypeBypass)
      .value("Num", mapspace::Dimension::Num)
      .def_static("values", []() {
        return std::array{
            mapspace::Dimension::IndexFactorization,
            mapspace::Dimension::LoopPermutation, mapspace::Dimension::Spatial,
            mapspace::Dimension::DatatypeBypass, mapspace::Dimension::Num};
      });

  py::class_<mapspace::ID>(m, "ID")
      .def("__getitem__",
           [](mapspace::ID& id, unsigned idx) -> uint128_t { return id[idx]; })
      .def("__getitem__",
           [](mapspace::ID& id, mapspace::Dimension dim) {
             return id[unsigned(dim)];
           })
      .def("__setitem__", [](mapspace::ID& id, mapspace::Dimension dim,
                             uint128_t v) { id.Set(unsigned(dim), v); })
      .def("__setitem__",
           [](mapspace::ID& id, unsigned idx, uint128_t v) { id.Set(idx, v); });

  py::class_<mapspace::Status>(m, "Status")
      .def_readonly("success", &mapspace::Status::success)
      .def_readonly("fail_reason", &mapspace::Status::fail_reason)
      .def("__str__", &StatusRepr)
      .def("__repr__", &StatusRepr);

  py::class_<mapspace::MapSpace, PyMapSpace>(m, "MapSpace")
      .def_static("parse_and_construct", &mapspace::ParseAndConstruct,
                  py::call_guard<py::scoped_ostream_redirect,
                                 py::scoped_estream_redirect>());

  py::class_<mapspace::Uber, mapspace::MapSpace>(m, "Uber")
      .def(py::init<config::CompoundConfigNode, config::CompoundConfigNode,
                    model::Engine::Specs, const problem::Workload&, bool>(),
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("split", &mapspace::Uber::Split,
           py::return_value_policy::reference_internal,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("construct_mapping",
           [](mapspace::Uber& m, mapspace::ID mapping_id,
              bool break_on_failure) {
             Mapping mapping;
             auto construction_status =
                 m.ConstructMapping(mapping_id, &mapping, break_on_failure);
             return std::make_tuple(construction_status, mapping);
           })
      .def("size", [](mapspace::Uber& m) { return m.Size(); })
      .def("size", [](mapspace::Uber& m,
                      mapspace::Dimension dim) { return m.Size(dim); })
      .def("all_sizes", &mapspace::Uber::AllSizes);
}

}  // namespace pytimeloop::mapspace_bindings
