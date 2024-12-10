from collections import defaultdict
from enum import auto, Flag
from functools import reduce
from operator import or_

from pytimeloop.fastfusion.pareto import MAPPING, nameloop2col
from pytimeloop.fastfusion.sim import TensorStorage, Tiling, Loop

from pytimeloop.looptree.energy import gather_actions, get_accesses
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups


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

    if Metrics.LATENCY in metrics:
        results["Latency"] = result.temporal_steps[einsum_id]

    if Metrics.ENERGY in metrics:
        results["Energy"] = energy

    offchip_accesses = 0
    for (level, tensor, einsum), count in accesses.items():
        if level == 0:
            offchip_accesses += count
        fulltiling.append(f"Ac_{level}_{tensor}={count:.2e}")
    if Metrics.OFF_CHIP_ACCESSES in metrics:
        results["Offchip_Ac"] = offchip_accesses

    fulltiling.append(f"{result.fanout}")

    if Metrics.OCCUPANCY in metrics:
        for (storage_id, n_loops), size in reservations.items():
            key = nameloop2col(storage_id, n_loops)
            results.setdefault(key, 0)
            results[key] += size
        for r in results:
            if "RESOURCE" in r:
                fulltiling.append(f"{r.replace('RESOURCE', 'R')}={results[r]:.2e}")

    if Metrics.LATENCY in metrics:
        fulltiling.append(f"L={results['Latency']:.2e}")

    if Metrics.ENERGY in metrics:
        fulltiling.append(f"E={results['Energy']:.2e}")

    results[MAPPING] = {einsum_id: str(fulltiling)}

    if Metrics.OP_INTENSITY in metrics:
        results["Op_Intensity"] = result.op_intensity[1]
    
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

