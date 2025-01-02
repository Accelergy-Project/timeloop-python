from collections import defaultdict
from enum import auto, Flag
from functools import reduce
from operator import or_

from pytimeloop.fastfusion.pareto import LOGSTRING, MAPPING, nameloop2col, DICT_COLUMNS
from pytimeloop.fastfusion.sim import TensorStorage, Tiling, Loop

from pytimeloop.looptree.energy import gather_actions, get_accesses
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups

import pytimeloop.fastfusion.looptreedisplay as looptreedisplay


class Metrics(Flag):
    LATENCY = auto()
    ENERGY = auto()
    OCCUPANCY = auto()
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
    logfunc=None,
    metrics=Metrics.all_metrics()
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
    reservations_logstring = []
    found_intermediate_tensors = 0
    
    def record_backing_storage(dspace, target, n_loops):
        # Returns true if it's the backing storage
        if dspace in found_tensors:
            return False
        
        nonlocal found_intermediate_tensors
        tensors.append(TensorStorage(dspace, target, n_loops, result.occupancy[(target, dspace)]))
        found_tensors.append(dspace)
        if dspace in intermediate_tensors:
            found_intermediate_tensors += 1
        return True

    def record_non_backing_reservation(dspace, target, n_loops):
        reservations.setdefault((target, n_loops), 0)
        reservations[(target, n_loops)] += result.occupancy[(target, dspace)]

    logstring = []
    full_tiling = []
    all_storages = []
    for node in mapping:
        if node["type"] == "storage":
            storages = []
            for dspace in node["dspace"]:
                if not record_backing_storage(dspace, node["target"], len(cur_loops)):
                    record_non_backing_reservation(dspace, node["target"], len(cur_loops))
                storages.append(TensorStorage(dspace, node["target"], len(cur_loops), result.occupancy[(node["target"], dspace)]))
            reservations_logstring += storages
            all_storages += storages
            logstring.append(f"Strg({node['dspace']} in {node['target']})")

        elif node["type"] == "spatial" or node["type"] == "temporal":
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

            if found_intermediate_tensors < len(intermediate_tensors):
                cur_loops.append(loop)
            full_tiling.append(loop)
            logstring.append(f"{node['type'][0].upper()}{node['rank']} size {tile_shape}")

    n_loops_of_intermediates = set()
    for t in tensors:
        if t.tensor_id not in intermediate_tensors:
            continue
        n_loops_of_intermediates.add(t.above_loop_index)
    if len(n_loops_of_intermediates) > 1 and not explore_fusion_uneven:
        logfunc(f"n_loops_of_intermediates: {n_loops_of_intermediates}")

    tiling_compatibility = Tiling(
        loops=tuple(cur_loops),
        tensors=frozenset(t for t in tensors if t.tensor_id in intermediate_tensors),
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

    if Metrics.OCCUPANCY in metrics:
        for (storage_id, n_loops), size in reservations.items():
            key = nameloop2col(storage_id, n_loops)
            results.setdefault(key, 0)
            results[key] += size

    if Metrics.LATENCY in metrics:
        logstring.append(f"L={results['Latency']:.2e}")

    if Metrics.ENERGY in metrics:
        logstring.append(f"E={results['Energy']:.2e}")
    
    for r in reservations_logstring:
        r: TensorStorage
        key = nameloop2col(r.backer_id, r.above_loop_index)
        results.setdefault(key, 0)
        results[key] += size

    results[LOGSTRING] = {einsum_id: str(logstring)}
    results[MAPPING] = {einsum_id: tiling_full}

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

