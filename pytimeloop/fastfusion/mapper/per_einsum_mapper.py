from collections import defaultdict
from functools import reduce
from operator import or_

from pytimeloop.fastfusion.fastmodel import compile_mapping
from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.mapper.logging import log_worker
from pytimeloop.fastfusion.pareto import nameloop2col
from pytimeloop.fastfusion.pareto import MAPPING
from pytimeloop.fastfusion.sim import TensorStorage, Tiling, Loop
from .per_einsum_subspaces.subspaces import (
    LinearMapping,
    make_temporal_fors,
    make_spatial_fors,
    make_storage,
    explore_tile_shape
)

from pytimeloop.looptree.energy import gather_actions, get_accesses
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer


@log_worker(f"{__name__}:_mapper_place_fusion_level")
def mapper_place_fusion_level(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    spec,
    explore_glb_uneven,
    explore_pe_uneven,
    einsum_id,
    energy_dict,
    partial_mapping,
    log_queue=None,
    verbose_stream=None,
):
    # if log_queue is not None:
    #     log_queue.info(f"[{einsum_id}] Exploring mapspace of Einsum {einsum_id}")
    logfunc = lambda msg: None # log_queue.debug(f"[{einsum_id}] " + msg)

    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    einsum_id_to_name = workload.einsum_id_to_name()
    rank_name_to_id = workload.dimension_name_to_id()
    tensor_name_to_id = workload.data_space_name_to_id()

    mac_parallel_shape = mac_array_constraint.array_shape_in_parallel_dimension
    mac_reduced_shape = mac_array_constraint.array_shape_in_reduced_dimension

    einsum_name_to_parallel_rank_name = mac_array_constraint.parallel_rank
    einsum_name_to_reduced_rank_name = mac_array_constraint.reduced_rank

    bindings, max_fanout, max_capacity = get_hardware_levels(spec.architecture)

    data = defaultdict(list)
    tensors = workload.tensors_read_by_einsum(einsum_id) \
            | workload.tensors_written_by_einsum(einsum_id)
    intermediate_tensors = tensors & get_intermediate_tensors(workload)

    einsum_name = einsum_id_to_name[einsum_id]
    mac_parallel_rank_name = einsum_name_to_parallel_rank_name[einsum_name]
    mac_parallel_rank_id = rank_name_to_id[mac_parallel_rank_name]
    mac_reduced_rank_name = einsum_name_to_reduced_rank_name[einsum_name]
    mac_reduced_rank_id = rank_name_to_id[mac_reduced_rank_name]

    weight_tensor_name = mac_array_constraint.weight_tensor[einsum_name]
    weight_tensor_id = tensor_name_to_id[weight_tensor_name]
    weight_ranks = analyzer.einsum_dims_relevant_to_tensor(einsum_id, weight_tensor_id)
    other_weight_ranks = weight_ranks - {mac_parallel_rank_id, mac_reduced_rank_id}
    all_ranks = workload.einsum_ospace_dimensions(einsum_id)
    non_weight_ranks = set(all_ranks) - weight_ranks

    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    einsum_shape = {
        rank_id: workload.get_rank_shape(rank_id)[1] + 1 for rank_id in all_ranks
    }
    count = 0

    for partial_mapping in make_temporal_fors(  # PE temporal
        partial_mapping, all_ranks
    ):
        # No bypassing at PE level. Can relax to explore more mappings
        pe_must_retain = tensors
        pe_can_retain = set()
        for partial_mapping in make_storage(  # PE storage
            partial_mapping,
            level=2,  # PE level
            must_retain_tensors=pe_must_retain,
            can_retain_tensors=pe_can_retain,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=explore_pe_uneven,
        ):
            for partial_mapping in make_mac_level_loops(
                partial_mapping,
                einsum_id,
                mac_parallel_rank_id,
                mac_parallel_shape,
                mac_reduced_rank_id,
                mac_reduced_shape,
                non_weight_ranks,
                other_weight_ranks,
            ):
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
                    is_pareto, results, fulltiling = process_result(
                        res,
                        shape,
                        data,
                        einsum_id,
                        intermediate_tensors,
                        partial_mapping,
                        bindings,
                        workload,
                        energy_dict,
                        equivalent_groups,
                        logfunc=logfunc,
                        explore_fusion_uneven=explore_glb_uneven
                    )
                    if count % 1e4 == 0:
                        print(f"Einsum {einsum_id} #{count}, fulltiling: {fulltiling}")
                    # shape_subspace.register_result(is_pareto, results)
    return einsum_id, data, count


@log_worker(f"{__name__}:_get_top_loop_jobs")
def get_top_loop_jobs(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    spec,
    explore_glb_uneven,
    explore_pe_uneven,
    einsums_to_explore,
    energy_dict,
    log_queue=None,
    verbose_stream=None,
):
    args = []
    for einsum_id in einsums_to_explore:
        if log_queue is not None:
            log_queue.info(f"[{einsum_id}] Exploring mapspace of Einsum {einsum_id}")
            logfunc = lambda msg: log_queue.debug(f"[{einsum_id}] " + msg)
        else:
            logfunc = lambda msg: None  # do nothing

        workload = LooptreeWorkload.parse_cfg(config.root["problem"])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

        data = {}
        data[einsum_id] = defaultdict(lambda: defaultdict(lambda: list()))
        tensors = workload.tensors_read_by_einsum(einsum_id) \
                | workload.tensors_written_by_einsum(einsum_id)
        intermediate_tensors = tensors & get_intermediate_tensors(workload)
        all_ranks = workload.einsum_ospace_dimensions(einsum_id)


        tensor_to_relevant_ranks = {
            tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
            for tensor in tensors
        }

        top_level_ranks = reduce(
            or_, (tensor_to_relevant_ranks[t] for t in intermediate_tensors), set()
        )

        mapping = LinearMapping()
        logfunc(f"Allowed top-level loop ranks: {top_level_ranks}")

        off_chip_must_retain = tensors - intermediate_tensors
        off_chip_can_retain = intermediate_tensors
        for partial_mapping in make_storage(  # Off-chip level
            mapping,
            level=0,
            must_retain_tensors=off_chip_must_retain,
            can_retain_tensors=off_chip_can_retain,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=False,
            add_split_at_tensors=intermediate_tensors,
        ):
            for partial_mapping in make_temporal_fors(  # GLB temporal
                partial_mapping,
                top_level_ranks,
            ):
                glb_must_retain = set(intermediate_tensors)
                glb_can_retain = set()
                for partial_mapping in make_storage(  # GLB level
                    partial_mapping,
                    level=1,
                    must_retain_tensors=glb_must_retain,
                    can_retain_tensors=glb_can_retain,
                    must_fully_reuse_tensors=set(intermediate_tensors),
                    tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                    explore_uneven=explore_glb_uneven,
                    add_split_at_tensors=intermediate_tensors,
                    must_have_terminal_storage=True,  # GLB only opt.
                    logfunc=None
                ):
                    for partial_mapping in make_spatial_fors(  # PE spatial
                        partial_mapping,
                        all_ranks,
                        max_factor=pe_array_constraint.array_shape
                    ):
                        args.append(dict(
                            config=config,
                            pe_array_constraint=pe_array_constraint,
                            mac_array_constraint=mac_array_constraint,
                            spec=spec,
                            explore_glb_uneven=explore_glb_uneven,
                            explore_pe_uneven=explore_pe_uneven,
                            einsum_id=einsum_id,
                            energy_dict=energy_dict,
                            partial_mapping=partial_mapping,
                            log_queue=log_queue,
                            verbose_stream=verbose_stream,
                        ))
    return args


def make_mac_level_loops(
    mapping,
    einsum_id,
    parallel_rank,
    parallel_rank_shape,
    reduced_rank,
    reduced_rank_shape,
    non_weight_ranks,
    other_weight_ranks,
):
    mapping = mapping.copy()
    for rank in other_weight_ranks:
        mapping.add_temporal(rank, 1)
    mapping.add_temporal(parallel_rank, parallel_rank_shape)
    mapping.add_temporal(reduced_rank, reduced_rank_shape)
    for rank in non_weight_ranks:
        mapping.add_temporal(rank, 1)
    mapping.add_spatial(parallel_rank, 1)
    mapping.add_spatial(reduced_rank, 1)
    mapping.add_compute(einsum_id, 3)
    yield mapping


def process_result(
    result,
    shape,
    compatibility_to_df,
    einsum_id,
    intermediate_tensors,
    mapping,
    bindings,
    workload,
    energy_dict,
    equiv_groups: EquivalentGroups,
    explore_fusion_uneven,
    logfunc=None,
):
    actions = gather_actions(
        result, {"type": "fused", "nodes": mapping}, workload, bindings, is_path=True
    )
    accesses = defaultdict(lambda: 0)
    reads, writes = get_accesses(
        result, {"type": "fused", "nodes": mapping}, workload, is_path=True
    )
    for k, v in reads.items():
        accesses[k] += v
    for k, v in writes.items():
        accesses[k] += v

    energy = sum(
        energy_dict[comp_action] * counts for comp_action, counts in actions.items()
    )

    cur_idx = 0
    cur_loops = []
    tensors = []
    found_tensors = []
    reservations = {}
    found_intermediate_tensors = 0
    
    def record_backing_storage(dspace, target, n_loops):
        if dspace in found_tensors:
            return
        
        nonlocal found_intermediate_tensors
        tensors.append(TensorStorage(dspace, target, n_loops, 0))
        found_tensors.append(dspace)
        if dspace in intermediate_tensors:
            found_intermediate_tensors += 1

    def record_reservation(dspace, target, n_loops):
        reservations.setdefault((target, n_loops), 0)
        reservations[(target, n_loops)] += result.occupancy[(target, dspace)]


    fulltiling = []
    for node in mapping:
        if node["type"] == "storage":
            for dspace in node["dspace"]:
                record_backing_storage(dspace, node["target"], len(cur_loops))
                record_reservation(dspace, node["target"], len(cur_loops))
            fulltiling.append(f"Strg({node['dspace']} in {node['target']})")

        elif node["type"] == "spatial" or node["type"] == "temporal":
            if "tile_shape" in node:
                tile_shape = node["tile_shape"]
            else:
                tile_shape = shape[cur_idx]
                cur_idx += 1

            if found_intermediate_tensors < len(intermediate_tensors):
                cur_loops.append(
                    Loop(
                        str(equiv_groups.rank_to_group_id[node["rank"]]),
                        tile_shape,
                        node["type"] == "spatial",
                    )
                )
            fulltiling.append(f"{node['type'][0].upper()}{node['rank']} size {tile_shape}")

    n_loops_of_intermediates = set()
    for t in tensors:
        if t.tensor_id not in intermediate_tensors:
            continue
        n_loops_of_intermediates.add(t.above_loop_index)
    if len(n_loops_of_intermediates) > 1 and not explore_fusion_uneven:
        logfunc(f"n_loops_of_intermediates: {n_loops_of_intermediates}")

    tiling = Tiling(
        loops=tuple(cur_loops),
        tensors=frozenset(t for t in tensors if t.tensor_id in intermediate_tensors),
    )

    results = {}
    # results["Latency"] = result.temporal_steps[einsum_id]
    # results["Energy"] = energy
    offchip_accesses = 0
    for (level, tensor, einsum), count in accesses.items():
        if level == 0:
            offchip_accesses += count
        fulltiling.append(f"Ac_{level}_{tensor}={count:.2e}")
    results["Offchip_Ac"] = offchip_accesses
    fulltiling.append(f"{result.fanout}")
    for (storage_id, n_loops), size in reservations.items():
        key = nameloop2col(storage_id, n_loops)
        results.setdefault(key, 0)
        results[key] += size
    for r in results:
        if "RESOURCE" in r:
            fulltiling.append(f"{r.replace('RESOURCE', 'R')}={results[r]:.2e}")
    # fulltiling.append(f"L={results['Latency']:.2e}")
    # fulltiling.append(f"E={results['Energy']:.2e}")
    results[MAPPING] = {einsum_id: str(fulltiling)}
    
    is_pareto = True
    for prev_stats in compatibility_to_df[tiling]:
        keys = [k for k in results if k != MAPPING]
        if all(prev_stats.get(k, 0) <= results[k] for k in keys) and \
                any(prev_stats.get(k, 0) < results[k] for k in keys):
            is_pareto = False
            break
    if is_pareto:
        compatibility_to_df[tiling].append(results)
    results_return = {
        k: v for k, v in results.items() if k != MAPPING
    }
    return is_pareto, results_return, fulltiling


def get_hardware_levels(arch):
    bindings = {}
    fanout = {}
    max_capacity = {}
    for node in arch["nodes"]:
        bindings_id = len(bindings)
        bindings[bindings_id] = node["name"]
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
        attribute = node.attributes
        if "width" in attribute and "depth" in attribute:
            width = attribute.width
            depth = attribute.depth
            datawidth = attribute.datawidth
            if all(x is not None for x in (width, depth, datawidth)):
                max_capacity[bindings_id] = (
                    attribute.width * attribute.depth / attribute.datawidth
                )
    return bindings, fanout, max_capacity


def count(it):
    count = 0
    for _ in it:
        count += 1
    return count
