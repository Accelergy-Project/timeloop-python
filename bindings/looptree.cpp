#define LOOPTREE_SUPPORT

#ifdef LOOPTREE_SUPPORT

#include "pytimeloop/bindings/looptree.h"

#include <applications/looptree-model/model.hpp>
#include <workload/fused-workload.hpp>

#include <pybind11/stl.h>


#define FUSED_WORKLOAD_METHOD(python_name, cpp_name) \
    def(#python_name, &problem::FusedWorkload::cpp_name)

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
        .FUSED_WORKLOAD_METHOD(tensors_read_by_einsum, TensorsReadByEinsum)
        .FUSED_WORKLOAD_METHOD(tensors_written_by_einsum, TensorsWrittenByEinsum)
        .FUSED_WORKLOAD_METHOD(reader_einsums, ReaderEinsums)
        .FUSED_WORKLOAD_METHOD(writer_einsums, WriterEinsum)
        .FUSED_WORKLOAD_METHOD(einsum_dim_is_relevant_to_tensor,
                               EinsumDimIsRelevantToTensor)
        .FUSED_WORKLOAD_METHOD(einsum_dims_relevant_to_tensor,
                               EinsumDimsRelevantToTensor)
        .def_static("parse_cfg", &problem::ParseFusedWorkload);
  }
}

#endif
