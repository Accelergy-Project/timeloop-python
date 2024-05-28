#include "pytimeloop/bindings/applications.h"

#include <applications/mapper/mapper.hpp>
#include <applications/model/model.hpp>


namespace py = pybind11;

namespace pytimeloop::application_bindings
{
  void BindModelApplication(py::module& m);
  void BindMapperApplication(py::module& m);

  void BindApplications(py::module& m)
  {
    BindModelApplication(m);
    BindMapperApplication(m);
  }

  void BindModelApplication(py::module& m)
  {
    py::class_<application::Model>(m, "ModelApp")
        .def(py::init<config::CompoundConfig*, std::string, std::string>())
        .def("run", &application::Model::Run);
  }

  void BindMapperApplication(py::module& m)
  {
    py::class_<application::Mapper>(m, "MapperApp")
        .def(py::init<config::CompoundConfig*, std::string, std::string>())
        .def("run", &application::Mapper::Run);
  }
}