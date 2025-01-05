from collections import defaultdict
from enum import auto, Flag
from functools import reduce
from operator import or_

from pytimeloop.fastfusion.pareto import LOGSTRING, MAPPING, STATS, nameloop2col, DICT_COLUMNS, RESERVED_COLUMNS
from pytimeloop.fastfusion.sim import TensorStorage, Tiling, Loop

from pytimeloop.looptree.energy import gather_actions, get_accesses
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups

import pytimeloop.fastfusion.looptreedisplay as looptreedisplay


class Metrics(Flag):
    LATENCY = auto()
    ENERGY = auto()
    # OCCUPANCY = auto()
    OFF_CHIP_ACCESSES = auto()
    OP_INTENSITY = auto()

    @classmethod
    def all_metrics(cls):
        return reduce(or_, iter(cls), cls.LATENCY) ^ Metrics.OP_INTENSITY


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
    einsum_shape,
    logfunc=None,
    metrics=Metrics.all_metrics(),
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
    intermediate_backing_storages = []
    non_intermediate_or_non_backing_storages = []
    all_storages = []
    intermediates_to_find = set(intermediate_tensors)
    n_steps = 1
    ranks_remaining = {k: v for k, v in einsum_shape.items()}
    n_repititions = 1
    
    def record_storage(node):
        for dspace in node["dspace"]:
            storage = TensorStorage(
                dspace, node["target"], 
                len(full_tiling), 
                result.occupancy[(node["target"], dspace)],
                n_repititions=n_repititions,
            )
            all_storages.append(storage)
            t = storage.tensor_id
            if t in intermediates_to_find:
                intermediates_to_find.remove(t)
                intermediate_backing_storages.append(storage)
            else:
                non_intermediate_or_non_backing_storages.append(storage)
        logstring.append(f"Strg({node['dspace']} in {node['target']})")
            
    def record_loop(node):
        nonlocal cur_idx, n_repititions
        if "tile_shape" in node:
            tile_shape = node["tile_shape"]
        else:
            tile_shape = shape[cur_idx]
            cur_idx += 1
        loop = Loop(
            str(equiv_groups.rank_to_group_id[node["rank"]]),
            tile_shape,
            node["type"] == "spatial",
        )
        n_repititions *= (ranks_remaining[node["rank"]] // tile_shape)
        ranks_remaining[node["rank"]] = tile_shape
        if intermediates_to_find:
            cur_loops.append(loop)
        full_tiling.append(loop)
        logstring.append(f"{node['type'][0].upper()}{node['rank']} size {tile_shape}")


    logstring = []
    full_tiling = []
    for node in mapping:
        if node["type"] == "storage":
            record_storage(node)
        elif node["type"] == "spatial" or node["type"] == "temporal":
            record_loop(node)

    tiling_compatibility = Tiling(
        loops=tuple(cur_loops),
        tensors=frozenset(t for t in intermediate_backing_storages if t.tensor_id in intermediate_tensors),
    )
    tiling_full = Tiling(
        loops=tuple(full_tiling),
        tensors=frozenset(all_storages),
    )

    results = {}

    if Metrics.LATENCY in metrics:
        results["Latency"] = result.temporal_steps[einsum_id]

    if Metrics.ENERGY in metrics:
        results["Energy"] = energy

    offchip_accesses = 0
    for (level, tensor, einsum), count in accesses.items():
        if level == 0:
            offchip_accesses += count
        logstring.append(f"Ac_{level}_{tensor}={count:.2e}")

    if Metrics.OFF_CHIP_ACCESSES in metrics:
        results["Offchip_Ac"] = offchip_accesses

    logstring.append(f"{result.fanout}")

    # Handled below. All occupancies are needed to ensure occupancies
    # < capacity when we fuse
    # if Metrics.OCCUPANCY in metrics:
    #     for (storage_id, n_loops), size in non_intermediate_or_non_backing_storages.items():
    #         key = nameloop2col(storage_id, n_loops)
    #         results.setdefault(key, 0)
    #         results[key] += size

    # Only record non-backing reservations. We'll reserve backing storage later
    # when we free the tensors & we know all operations for which the tensor must
    # be backed
    for r in non_intermediate_or_non_backing_storages:
        r: TensorStorage
        key = nameloop2col(r.backer_id, r.above_loop_index)
        results.setdefault(key, 0)
        results[key] += r.tile_size
        # logstring.append(f"{r}")

    if Metrics.LATENCY in metrics:
        logstring.append(f"L={results['Latency']:.2e}")

    if Metrics.ENERGY in metrics:
        logstring.append(f"E={results['Energy']:.2e}")
    
    logstring.append(f"Results: {results}")
    results[LOGSTRING] = {einsum_id: str(logstring)}
    results[MAPPING] = {einsum_id: tiling_full}
    results[STATS] = {einsum_id: {k: v for k, v in results.items() if k not in RESERVED_COLUMNS}}
    

    if Metrics.OP_INTENSITY in metrics:
        results["Op_Intensity"] = result.op_intensity[1]
    
    is_pareto = True
    for prev_stats in compatibility_to_df[tiling_compatibility]:
        keys = [k for k in results if k not in DICT_COLUMNS]
        if all(prev_stats.get(k, 0) <= results[k] for k in keys) and \
                any(prev_stats.get(k, 0) < results[k] for k in keys):
            is_pareto = False
            break
    if is_pareto:
        compatibility_to_df[tiling_compatibility].append(results)
    results_return = {
        k: v for k, v in results.items() if k != LOGSTRING
    }
    return is_pareto, results_return, logstring

