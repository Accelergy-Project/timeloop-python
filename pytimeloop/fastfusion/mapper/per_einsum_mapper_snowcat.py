from copy import deepcopy
from collections import defaultdict

from joblib import delayed

from combinatorics.dependent_product import dependent_product
from combinatorics.splitter import split_dependent_product

from pytimeloop.fastfusion.fastmodel import compile_mapping
from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.mapper.per_einsum_mapper import explore_tile_shape, process_result, get_hardware_levels
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.snowcat import make_subspaces
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.snowcat_ffmt import make_ffmt_subspaces
from pytimeloop.fastfusion.util import parallel
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors
from pytimeloop.fastfusion.mapper.process_results import Metrics, process_result

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer


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
):
    data = {}
    for einsum_id in einsums_to_explore:
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
        #         for t in ["spatial"]: # "temporal", TEMPORAL DOESN"T WORK. WEIRD INTERACTIONS WITH LOOP RELEVANCE PRINCIPLE
        #             if p["type"] == t and prev["type"] == t and p["rank"] < prev["rank"]:
        #                 fail = True
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
                        rank_id_to_name={v: k for k, v in rank_name_to_id.items()},
                        tensor_id_to_name={v: k for k, v in tensor_name_to_id.items()},
                    )
            return result

        # for pm in partial_mappings:
        #     per_worker_exploration(*pm)
        results = parallel(delayed(per_worker_exploration)(*pm) for pm in partial_mappings)

        data[einsum_id] = defaultdict(list)
        for res in results:
            for k, v in res.items():
                data[einsum_id][k] += v

    return data

