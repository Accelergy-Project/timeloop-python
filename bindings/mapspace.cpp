#include "bindings.h"

// Timeloop headers
#include "mapspaces/mapspace-base.hpp"
#include "mapspaces/mapspace-factory.hpp"

void BindMapSpaceClasses(py::module& m) {
  py::class_<mapspace::MapSpace>(m, "NativeMapSpace")
      .def(py::init<model::Engine::Specs, const problem::Workload&>())
      .def("split", &mapspace::MapSpace::Split)
      .def_static("parse_and_construct", &mapspace::ParseAndConstruct);

  py::class_<mapspace::ID>(m, "NativeMapSpaceID")
      .def(py::init<std::array<uint128_t, int(mapspace::Dimension::Num)>>())
      .def("increment", &mapspace::ID::Increment)
      .def("read", &mapspace::ID::Read)
      .def("base", &mapspace::ID::Base)
      .def("__getitem__", &mapspace::ID::operator[])
      .def("set", &mapspace::ID::Set)
      .def("end_integer", &mapspace::ID::EndInteger)
      .def("integer", &mapspace::ID::Integer);
}
