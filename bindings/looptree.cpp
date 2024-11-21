#define LOOPTREE_SUPPORT

#ifdef LOOPTREE_SUPPORT

#include "pytimeloop/bindings/looptree.h"

#include <applications/looptree-model/model.hpp>
#include <workload/fused-workload.hpp>
#include <workload/fused-workload-dependency-analyzer.hpp>
#include <isl-wrapper/ctx-manager.hpp>

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
        .def(py::init<config::CompoundConfig*>())
        .def(py::init<const problem::FusedWorkload&, const mapping::FusedMapping&>())
        .def("run", &application::LooptreeModel::Run);

    py::class_<analysis::Temporal>(m, "TemporalTag").def(py::init<>());
    py::class_<analysis::Spatial>(m, "SpatialTag").def(py::init<int, analysis::BufferId>());
    py::class_<analysis::Sequential>(m, "SequentialTag").def(py::init<>());
    py::class_<analysis::PipelineTemporal>(m, "PipelineTemporalTag").def(py::init<>());
    py::class_<analysis::PipelineSpatial>(m, "PipelineSpatialTag").def(py::init<>());

    py::class_<application::LooptreeModel::Result>(m, "LooptreeResult")
        .def_readwrite("ops", &application::LooptreeModel::Result::ops)
        .def_readwrite("fills", &application::LooptreeModel::Result::fills)
        .def_readwrite("fills_by_parent", &application::LooptreeModel::Result::fills_by_parent)
        .def_readwrite("fills_by_peer", &application::LooptreeModel::Result::fills_by_peer)
        .def_readwrite("occupancy", &application::LooptreeModel::Result::occupancy)
        .def_readwrite("temporal_steps", &application::LooptreeModel::Result::temporal_steps);

    py::class_<mapping::FusedMapping>(m, "FusedMapping")
        .def_static("parse", &mapping::ParseMapping);

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
        .FUSED_WORKLOAD_METHOD(get_rank_shape, GetRankShape)
        .FUSED_WORKLOAD_METHOD(get_tensor_volume, GetTensorSize)
        .FUSED_WORKLOAD_METHOD(get_operation_space_volume, GetOspaceVolume)
        .def(py::pickle(
            [](const problem::FusedWorkload& workload)
            {
                using namespace problem;
                std::map<EinsumId, std::vector<DimensionId>>
                einsum_to_dimensions;
                std::map<EinsumId, std::set<DataSpaceId>>
                einsum_to_input_tensors;
                std::map<EinsumId, std::set<DataSpaceId>>
                einsum_to_output_tensors;
                std::map<std::pair<EinsumId, DataSpaceId>, std::string>
                einsum_dspace_to_projection;
                std::map<EinsumId, std::string> einsum_to_ospace_bound;
                for (const auto& [einsum_id, einsum_name] : workload.EinsumIdToName())
                {
                    einsum_to_dimensions[einsum_id] =
                        workload.EinsumOspaceDimensions(einsum_id);
                    einsum_to_input_tensors[einsum_id] =
                        workload.TensorsReadByEinsum(einsum_id);
                    einsum_to_output_tensors[einsum_id] =
                        workload.TensorsWrittenByEinsum(einsum_id);
                    for (const auto& dspace_id : einsum_to_input_tensors.at(einsum_id))
                    {
                        const auto& acc = workload.ReadAccessesAff(einsum_id, dspace_id);
                        std::stringstream ss;
                        ss << acc;
                        einsum_dspace_to_projection[std::make_pair(einsum_id, dspace_id)] =
                            ss.str();
                    }
                    for (const auto& dspace_id : einsum_to_output_tensors.at(einsum_id))
                    {
                        const auto& acc = workload.WriteAccessesAff(einsum_id, dspace_id);
                        std::stringstream ss;
                        ss << acc;
                        einsum_dspace_to_projection[std::make_pair(einsum_id, dspace_id)] =
                            ss.str();
                    }
                    const auto& bound = workload.EinsumOspaceBound(einsum_id);
                    std::stringstream ss;
                    ss << bound;
                    einsum_to_ospace_bound[einsum_id] = ss.str();
                }

                std::map<DataSpaceId, std::vector<DimensionId>>
                dspace_to_dimensions;
                std::map<DataSpaceId, std::string>
                dspace_to_bound;
                for (const auto& [dspace_id, dspace_name] : workload.DataSpaceIdToName())
                {
                    dspace_to_dimensions[dspace_id] =
                        workload.DataSpaceDimensions(dspace_id);

                    const auto& bound = workload.DataSpaceBound(dspace_id);
                    std::stringstream ss;
                    ss << bound;
                    dspace_to_bound[dspace_id] = ss.str();
                }

                return py::make_tuple(
                    workload.EinsumIdToName(),
                    workload.DataSpaceIdToName(),
                    workload.DimensionIdToName(),
                    einsum_to_dimensions,
                    einsum_to_input_tensors,
                    einsum_to_output_tensors,
                    einsum_dspace_to_projection,
                    einsum_to_ospace_bound,
                    dspace_to_dimensions,
                    dspace_to_bound
                );
            },
            [](py::tuple t)
            {
                using namespace problem;

                auto workload = std::make_unique<problem::FusedWorkload>();

                const auto einsum_id_to_name =
                    t[0].cast<std::map<EinsumId, std::string>>();
                const auto dspace_id_to_name =
                    t[1].cast<std::map<DataSpaceId, std::string>>();
                const auto dimension_id_to_name =
                    t[2].cast<std::map<DimensionId, std::string>>();
                for (const auto& [einsum_id, einsum_name] : einsum_id_to_name)
                {
                    workload->NewEinsum(einsum_name);
                }
                for (const auto& [dspace_id, dspace_name] : dspace_id_to_name)
                {
                    workload->NewDataSpace(dspace_name);
                }
                for (const auto& [dim_id, dim_name] : dimension_id_to_name)
                {
                    workload->NewDimension(dim_name);
                }

                const auto einsum_to_dimensions =
                    t[3].cast<std::map<EinsumId, std::vector<DimensionId>>>();
                const auto einsum_to_input_tensors =
                    t[4].cast<std::map<EinsumId, std::set<DataSpaceId>>>();
                const auto einsum_to_output_tensors =
                    t[5].cast<std::map<EinsumId, std::set<DataSpaceId>>>();
                const auto einsum_dspace_to_projection =
                    t[6].cast<std::map<std::pair<EinsumId, DataSpaceId>, std::string>>();
                const auto einsum_to_ospace_bound =
                    t[7].cast<std::map<EinsumId, std::string>>();
                const auto dspace_to_dimensions =
                    t[8].cast<std::map<DataSpaceId, std::vector<DimensionId>>>();
                const auto dspace_to_bound =
                    t[9].cast<std::map<DataSpaceId, std::string>>();
                for (const auto& [einsum_id, dims] : einsum_to_dimensions)
                {
                    for (const auto& dim_id : dims)
                    {
                        workload->AddDimToEinsumOspace(einsum_id, dim_id);
                    }
                }
                for (const auto& [dspace_id, dims] : dspace_to_dimensions)
                {
                    for (const auto& dim_id : dims)
                    {
                        workload->AddDimToDspace(dspace_id, dim_id);
                    }
                }
                for (const auto& [einsum_id, input_tensors] : einsum_to_input_tensors)
                {
                    for (const auto& input_tensor_id : input_tensors)
                    {
                        const auto& proj_str = einsum_dspace_to_projection.at(
                            std::make_pair(einsum_id, input_tensor_id)
                        );
                        auto proj = isl::multi_aff(GetIslCtx(), proj_str);
                        workload->SetEinsumProjection(einsum_id,
                                                      input_tensor_id,
                                                      false,
                                                      proj);
                    }
                }
                for (const auto& [einsum_id, output_tensors] : einsum_to_output_tensors)
                {
                    for (const auto& output_tensor_id : output_tensors)
                    {
                        const auto& proj_str = einsum_dspace_to_projection.at(
                            std::make_pair(einsum_id, output_tensor_id)
                        );
                        auto proj = isl::multi_aff(GetIslCtx(), proj_str);
                        workload->SetEinsumProjection(einsum_id,
                                                      output_tensor_id,
                                                      true,
                                                      proj);
                    }
                }
                for (const auto& [einsum_id, bound_str] : einsum_to_ospace_bound)
                {
                    auto bound = isl::set(GetIslCtx(), bound_str);
                    workload->SetEinsumOspaceBound(einsum_id, bound);
                }
                for (const auto& [dspace_id, bound_str] : dspace_to_bound)
                {
                    auto bound = isl::set(GetIslCtx(), bound_str);
                    workload->SetDataSpaceBound(dspace_id, bound);
                }

                return workload;
            }
        ))
        .def_static("parse_cfg", &problem::ParseFusedWorkload);

    py::class_<problem::FusedWorkloadDependencyAnalyzer>(m, "LooptreeWorkloadDependencyAnalyzer")
        .def(py::init<const problem::FusedWorkload&>())
        .FUSED_WORKLOAD_ANALYZER_METHOD(find_einsum_dependency_chain, FindEinsumDependencyChain)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dim_is_directly_relevant_to_tensor, EinsumDimIsDirectlyRelevantToTensor)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dim_is_relevant_to_tensor, EinsumDimIsRelevantToTensor)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dims_directly_relevant_to_tensor, EinsumDimsDirectlyRelevantToTensor)
        .FUSED_WORKLOAD_ANALYZER_METHOD(einsum_dims_relevant_to_tensor, EinsumDimsRelevantToTensor)
        .FUSED_WORKLOAD_ANALYZER_METHOD(equivalent_dimensions, EquivalentDimensions);
  }
}

#endif
