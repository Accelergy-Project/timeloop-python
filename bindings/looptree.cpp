#define LOOPTREE_SUPPORT

#ifdef LOOPTREE_SUPPORT

#include "pytimeloop/bindings/looptree.h"

#include <applications/looptree-model/model.hpp>
#include <workload/fused-workload.hpp>
#include <workload/fused-workload-dependency-analyzer.hpp>

#include <pybind11/stl.h>


#define FUSED_WORKLOAD_METHOD(python_name, cpp_name) \
    def(#python_name, &problem::FusedWorkload::cpp_name)

#define FUSED_WORKLOAD_ANALYZER_METHOD(python_name, cpp_name) \
    def(#python_name, &problem::FusedWorkloadDependencyAnalyzer::cpp_name)

namespace py = pybind11;


namespace pytimeloop::looptree_bindings
{
  void BindLooptree(py::module& m)
  {
    py::class_<application::LooptreeModel>(m, "LooptreeModelApp")
        .def(py::init<config::CompoundConfig*, std::string, std::string>())
        .def("run", &application::LooptreeModel::Run);

    py::class_<application::LooptreeModel::Result>(m, "LooptreeResult")
        .def_readwrite("ops", &application::LooptreeModel::Result::ops)
        .def_readwrite("fill", &application::LooptreeModel::Result::fill)
        .def_readwrite("occupancy", &application::LooptreeModel::Result::occupancy);

    py::class_<problem::FusedWorkload>(m, "LooptreeWorkload")
        .FUSED_WORKLOAD_METHOD(einsum_name_to_id, EinsumNameToId)
        .FUSED_WORKLOAD_METHOD(einsum_id_to_name, EinsumIdToName)
        .FUSED_WORKLOAD_METHOD(data_space_name_to_id, DataSpaceNameToId)
        .FUSED_WORKLOAD_METHOD(data_space_id_to_name, DataSpaceIdToName)
        .FUSED_WORKLOAD_METHOD(dimension_name_to_id, DimensionNameToId)
        .FUSED_WORKLOAD_METHOD(dimension_id_to_name, DimensionIdToName)
        .FUSED_WORKLOAD_METHOD(get_data_space_with_dim, GetDataSpaceWithDim)
        .FUSED_WORKLOAD_METHOD(get_einsum_with_dim, GetEinsumWithDim)
        .FUSED_WORKLOAD_METHOD(data_space_dimensions, DataSpaceDimensions)
        .FUSED_WORKLOAD_METHOD(einsum_ospace_dimensions, EinsumOspaceDimensions)
        .FUSED_WORKLOAD_METHOD(tensors_read_by_einsum, TensorsReadByEinsum)
        .FUSED_WORKLOAD_METHOD(tensors_written_by_einsum, TensorsWrittenByEinsum)
        .FUSED_WORKLOAD_METHOD(reader_einsums, ReaderEinsums)
        .FUSED_WORKLOAD_METHOD(writer_einsum, WriterEinsum)
        .def_static("parse_cfg", &problem::ParseFusedWorkload);

    py::class_<problem::FusedWorkloadDependencyAnalyzer>(m, "LooptreeWorkloadDependencyAnalyzer")
        .def(py::init<const problem::FusedWorkload&>())
        .FUSED_WORKLOAD_ANALYZER_METHOD(find_einsum_dependency_chain, FindEinsumDependencyChain)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dim_is_directly_relevant_to_tensor, EinsumDimIsDirectlyRelevantToTensor)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dim_is_relevant_to_tensor, EinsumDimIsRelevantToTensor)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dims_directly_relevant_to_tensor, EinsumDimsDirectlyRelevantToTensor)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dims_relevant_to_tensor, EinsumDimsRelevantToTensor);
  }
}

#endif
