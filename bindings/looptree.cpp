#define LOOPTREE_SUPPORT

#ifdef LOOPTREE_SUPPORT

#include "pytimeloop/bindings/looptree.h"
#include <applications/looptree-model/model.hpp>

namespace py = pybind11;

namespace pytimeloop::looptree_bindings
{
  void BindLooptree(py::module& m)
  {
    py::class_<application::LooptreeModel>(m, "LooptreeModelApp")
        .def(py::init<config::CompoundConfig*, std::string, std::string>());

    py::class_<application::LooptreeModel::Result>(m, "LooptreeResult")
        .def_readwrite("ops", &application::LooptreeModel::Result::ops)
        .def_readwrite("fill", &application::LooptreeModel::Result::fill)
        .def_readwrite("occupancy", &application::LooptreeModel::Result::occupancy);
  }
}

#endif
