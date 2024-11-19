from collections import defaultdict
from collections.abc import Callable, Set
from itertools import combinations, product, permutations
from functools import reduce
from operator import or_, mul

from combinatorics.dependent_product import dependent_product
from combinatorics.splitter import split_dependent_product

from pytimeloop.fastfusion.fastmodel import compile_mapping
from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.mapper.per_einsum_mapper import LinearMapping, make_storage, make_temporal_fors, make_spatial_fors, make_mac_level_loops, explore_tile_shape, process_result, get_hardware_levels
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer


def per_einsum_mapper_snowcat(
    config,
    spec,
    explore_glb_uneven,
    einsums_to_explore,
    energy_dict,
):
    for einsum_id in einsums_to_explore:
        workload = LooptreeWorkload.parse_cfg(config.root["problem"])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
        equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

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

        top_level_ranks = reduce(
            or_, (tensor_to_relevant_ranks[t] for t in intermediate_tensors), set()
        )

        data = {}
        data[einsum_id] = defaultdict(list)

        mapping = LinearMapping()

        def off_chip_storage(mapping):
            off_chip_must_retain = tensors - intermediate_tensors
            off_chip_can_retain = intermediate_tensors
            yield from make_storage(
                mapping,
                level=0,
                must_retain_tensors=off_chip_must_retain,
                can_retain_tensors=off_chip_can_retain,
                tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                explore_uneven=False,
                add_split_at_tensors=intermediate_tensors,
                return_retained_tensors=True,
            )

        def fused_temporal_fors(mapping, unfused_tensors):
            for partial_mapping in make_temporal_fors(mapping, all_ranks):
                for partial_mapping in make_temporal_fors(mapping, all_ranks):
                    for partial_mapping in make_temporal_fors_with_smallest_tile(mapping, all_ranks):
                        yield partial_mapping, unfused_tensors

        def glb_storage(mapping, unfused_tensors):
            glb_fused_tensors = intermediate_tensors - unfused_tensors
            yield from make_storage(
                mapping,
                level=1,
                must_retain_tensors=tensors,
                can_retain_tensors=set(),
                tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                explore_uneven=True,
                add_split_at_tensors=glb_fused_tensors
            )

        def mac(mapping):
             mapping.add_compute(einsum_id, 2)
             yield mapping

        subspaces = [
            lambda: [LinearMapping()],
            off_chip_storage,
            fused_temporal_fors,
            glb_storage,
            mac
        ]

        n_jobs=32
        parallelized_spaces, task_spaces = \
            split_dependent_product(n_split_min=n_jobs, spaces=subspaces)

        Parallel(n_jobs=n_jobs)

        count = 0
        print(len(list(dependent_product(subspaces))))
        for partial_mapping in dependent_product(subspaces):
            _, compiled_results = compile_mapping(
                partial_mapping, workload, analyzer
            )
            tile_shape_explorer = explore_tile_shape(
                partial_mapping,
                einsum_shape,
                compiled_results,
                max_capacity,
                max_fanout,
            )
            # HACKY: Pop out the subspace object as the first in the iterator
            shape_subspace = next(tile_shape_explorer)

            for shape, res in tile_shape_explorer:
                count += 1
                is_pareto, fulltiling = process_result(
                    res,
                    shape,
                    data[einsum_id],
                    einsum_id,
                    intermediate_tensors,
                    partial_mapping,
                    bindings,
                    workload,
                    energy_dict,
                    equivalent_groups,
                    explore_fusion_uneven=explore_glb_uneven
                )
                # if is_pareto:
                #     shape_subspace.register_pareto_point()


def make_temporal_fors_with_smallest_tile(original, ranks):
    for r in range(len(ranks) + 1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_temporal(r, tile_shape=1)
            yield mapping