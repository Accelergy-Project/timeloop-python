#include "pytimeloop/bindings/applications.h"

#include <applications/mapper/mapper.hpp>
#include <applications/mapper/mapper-thread.hpp>
#include <applications/model/model.hpp>


namespace py = pybind11;

namespace pytimeloop::application_bindings
{
  void BindApplications(py::module& m)
  {
    py::class_<application::Model>(m, "ModelApp")
        .def(py::init<config::CompoundConfig*, std::string, std::string>())
        .def("run", &application::Model::Run);

    py::class_<application::Model::Stats>(m, "ModelResult")
        .def_readwrite("cycles", &application::Model::Stats::cycles)
        .def_readwrite("energy", &application::Model::Stats::energy);

    py::class_<application::Mapper>(m, "MapperApp")
        .def(py::init<config::CompoundConfig*, std::string, std::string>())
        .def("run", &application::Mapper::Run)
        .def("get_global_best", &application::Mapper::GetGlobalBest);

    // EvaluationResult in mapper-thread.cpp
    py::class_<EvaluationResult>(m, "MapperResult")
        .def_readwrite("valid", &EvaluationResult::valid)
        .def_readwrite("mapping", &EvaluationResult::mapping)
        .def_readwrite("stats", &EvaluationResult::stats);

  }
}