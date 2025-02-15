from copy import deepcopy
from collections import defaultdict
import itertools

from joblib import delayed

from combinatorics.dependent_product import dependent_product
from combinatorics.splitter import split_dependent_product
import pandas as pd

from pytimeloop.fastfusion.fastmodel import compile_mapping
from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.mapper.per_einsum_mapper import explore_tile_shape, process_result, get_hardware_levels
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.snowcat import make_subspaces
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.snowcat_ffmt import make_ffmt_subspaces
from pytimeloop.fastfusion.pareto import Pareto, makepareto
from pytimeloop.fastfusion.sim import SIM
from pytimeloop.fastfusion.util import parallel
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors
from pytimeloop.fastfusion.mapper.process_results import Metrics, process_result

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer


def _per_einsum_mapper_snowcat(
    config,
    spec,
    explore_glb_uneven,
    einsum_id,
    energy_dict,
    ffmt=False,
    ffmt_refetch_weights=True,
    dataflow_constraint=None,
    metrics=Metrics.all_metrics(),
    tag_with: tuple[callable] = (),
):
    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    einsum_id_to_name = workload.einsum_id_to_name()
    rank_name_to_id   = workload.dimension_name_to_id()
    tensor_name_to_id = workload.data_space_name_to_id()

    tensors = workload.tensors_read_by_einsum(einsum_id) \
            | workload.tensors_written_by_einsum(einsum_id)
    intermediate_tensors = tensors & get_intermediate_tensors(workload)
    all_ranks = workload.einsum_ospace_dimensions(einsum_id)

    bindings, max_fanout, max_capacity = get_hardware_levels(spec.architecture)

    all_ranks = workload.einsum_ospace_dimensions(einsum_id)

    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    einsum_shape = {
        rank_id: workload.get_rank_shape(rank_id)[1] + 1 for rank_id in all_ranks
    }


    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    if not ffmt:
        subspaces = make_subspaces(tensors,
                                    intermediate_tensors,
                                    tensor_to_relevant_ranks,
                                    einsum_id,
                                    workload,
                                    dataflow_constraint[einsum_id])
    else:
        subspaces = make_ffmt_subspaces(tensors,
                                        intermediate_tensors,
                                        tensor_to_relevant_ranks,
                                        einsum_id,
                                        workload,
                                        refetch_weights=ffmt_refetch_weights)

    n_jobs=32
    parallelized_spaces, task_spaces = \
        split_dependent_product(n_split_min=n_jobs, spaces=subspaces)

    partial_mappings = list(dependent_product(parallelized_spaces))
    partial_mappings = [x if isinstance(x, tuple) else (x,) for x in partial_mappings]
    rank_id_to_name = {v: k for k, v in rank_name_to_id.items()}
    tensor_id_to_name = {v: k for k, v in tensor_name_to_id.items()}
    input_tensors = set(tensor_id_to_name[t] for t in workload.tensors_read_by_einsum(einsum_id))
    output_tensors = set(tensor_id_to_name[t] for t in workload.tensors_written_by_einsum(einsum_id))
    rank_name_to_shared_name = {
        rank_id_to_name[k]: rank_id_to_name[v] for k, v in equivalent_groups.rank_to_group_id.items()
    }

    # successful_partial_mappings = []
    # for p in partial_mappings:
    #     partial_mapping = p[0]
    #     found_storages = set()
    #     fail = False
    #     for i, p in enumerate(partial_mapping):
    #         if p["type"] == "storage":
    #             for t in set(p["dspace"]) - found_storages:
    #                 for p2 in partial_mapping[:i]:
    #                     if p2["type"] in ["temporal", "spatial"] and p2["rank"] not in tensor_to_relevant_ranks[t]:
    #                         fail = True
    #                         break
    #             found_storages |= set(p["dspace"])
    #         if len(found_storages) < len(tensors) or i == 0:
    #             continue
    #         prev = partial_mapping[i - 1]
    #         for t in ["spatial"]: # "temporal", TEMPORAL DOESN"T WORK. WEIRD INTERACTIONS WITH LOOP RELEVANCE PRINCIPLEz
    #         if not fail:
    #             successful_partial_mappings.append(p)
    # partial_mappings = successful_partial_mappings

    def per_worker_exploration(*args):
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
        local_task_spaces = deepcopy(task_spaces)
        local_task_spaces[0] = lambda : task_spaces[0](*args)
        result = defaultdict(list)
        for partial_mapping in dependent_product(local_task_spaces):
            _, compiled_results = compile_mapping(
                partial_mapping, workload, analyzer
            )
            tile_shape_explorer = explore_tile_shape(
                partial_mapping,
                einsum_shape,
                compiled_results,
                max_capacity,
                max_fanout,
                tensors=tensors,
            )
            # HACKY: Pop out the subspace object as the first in the iterator
            shape_subspace = next(tile_shape_explorer)

            count = 0
            for shape, res in tile_shape_explorer:
                count += 1
                is_pareto, results, fulltiling = process_result(
                    res,
                    shape,
                    result,
                    einsum_id,
                    intermediate_tensors,
                    partial_mapping,
                    bindings,
                    workload,
                    energy_dict,
                    equivalent_groups,
                    explore_fusion_uneven=explore_glb_uneven,
                    einsum_shape=einsum_shape,
                    metrics=metrics,
                    einsum_id_to_name=einsum_id_to_name,
                    rank_id_to_name=rank_id_to_name,
                    tensor_id_to_name=tensor_id_to_name,
                    rank_name_to_shared_name=rank_name_to_shared_name,
                    input_tensors=input_tensors,
                    output_tensors=output_tensors,
                    tag_with=tag_with,
                )
        return einsum_id, {k: makepareto(pd.DataFrame(v).fillna(0)) for k, v in result.items()}


    # # for pm in partial_mappings:
    # #     per_worker_exploration(*pm)
    # data[einsum_id] = defaultdict(list)
    # for res in parallel(
    #     [delayed(per_worker_exploration)(*pm) for pm in partial_mappings],
    #     return_as_generator=True,
    #     pbar=f"Generating data for Einsum {einsum_id}. {i+1}/{len(einsums_to_explore)}",
    # ):
    #     for k, v in res.items():
    #         data[einsum_id][k[0]] += v

    return [delayed(per_worker_exploration)(*pm) for pm in partial_mappings]

def per_einsum_mapper_snowcat(
    config,
    spec,
    explore_glb_uneven,
    einsums_to_explore,
    energy_dict,
    ffmt=False,
    ffmt_refetch_weights=True,
    dataflow_constraint=None,
    metrics=Metrics.all_metrics(),
    tag_with: tuple[callable] = (),
):
    # return _per_einsum_mapper_snowcat(
    #     config,
    #     spec,
    #     explore_glb_uneven,
    #     einsums_to_explore,
    #     energy_dict,
    #     ffmt=ffmt,
    #     ffmt_refetch_weights=ffmt_refetch_weights,
    #     dataflow_constraint=dataflow_constraint,
    #     metrics=metrics,
    #     tag_with=tag_with,
    # )
    
    jobs = list(j for einsum_id in einsums_to_explore for j in _per_einsum_mapper_snowcat(
            config,
            spec,
            explore_glb_uneven,
            einsum_id,
            energy_dict,
            ffmt=ffmt,
            ffmt_refetch_weights=ffmt_refetch_weights,
            dataflow_constraint=dataflow_constraint,
            metrics=metrics,
            tag_with=tag_with,
        )
    )
    data = {einsum_id: defaultdict(list) for einsum_id in einsums_to_explore}

    for einsum_id, result in parallel(jobs, pbar="Generating Single-Einsum Mappings"):
        d = data[einsum_id]
        for k, v in result.items():
            d[k[0]].append(v)
            
    def makesim(einsum_id, tiling, data):
        return einsum_id, SIM(tiling, Pareto(pd.concat(data).fillna(0), skip_pareto=len(data) == 1))
            
    data2 = defaultdict(list)
    jobs = [delayed(makesim)(einsum_id, tiling, data) for einsum_id, tilings in data.items() for tiling, data in tilings.items()]
    for einsum_id, sim in parallel(jobs, pbar="Generating SIMs"):
        data2[einsum_id].append(sim)
    
    return data2