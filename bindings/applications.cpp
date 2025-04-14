#include "pytimeloop/bindings/applications.h"

#include <applications/mapper/mapper.hpp>
#include <applications/mapper/mapper-thread.hpp>
#include <applications/model/model.hpp>


#define PROPERTY(prop_name, class_name) \
    def_readwrite(#prop_name, &class_name::prop_name)


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
        .def_readwrite("energy", &application::Model::Stats::energy)
        .def_readwrite("stats_string", &application::Model::Stats::stats_string)
        .def_readwrite("map_string", &application::Model::Stats::map_string)
        .def_readwrite("xml_map_and_stats_string", &application::Model::Stats::xml_map_and_stats_string)
        .def_readwrite("tensella_string", &application::Model::Stats::tensella_string);

    py::class_<application::Mapper>(m, "MapperApp")
        .def(py::init<config::CompoundConfig*, std::string, std::string>())
        .def("run", &application::Mapper::Run)
        .def("get_global_best", &application::Mapper::GetGlobalBest);
    
    py::class_<application::Mapper::Result>(m, "MapperResult")
        .def_readwrite("mapping_cpp_string", &application::Mapper::Result::mapping_cpp_string)
        .def_readwrite("mapping_yaml_string", &application::Mapper::Result::mapping_yaml_string)
        .def_readwrite("mapping_string", &application::Mapper::Result::mapping_string)
        .def_readwrite("stats_string", &application::Mapper::Result::stats_string)
        .def_readwrite("tensella_string", &application::Mapper::Result::tensella_string)
        .def_readwrite("xml_mapping_stats_string", &application::Mapper::Result::xml_mapping_stats_string)
        .def_readwrite("orojenesis_string", &application::Mapper::Result::orojenesis_string);

    // EvaluationResult in mapper-thread.cpp
    py::class_<EvaluationResult>(m, "MapperEvaluationResult")
        .def_readwrite("valid", &EvaluationResult::valid)
        .def_readwrite("mapping", &EvaluationResult::mapping)
        .def_readwrite("stats", &EvaluationResult::stats);

  }
}