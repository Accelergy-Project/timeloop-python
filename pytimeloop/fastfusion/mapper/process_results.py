from collections import defaultdict
from enum import auto, Flag
from functools import reduce
from operator import or_

from pytimeloop.fastfusion.pareto import (
    LOGSTRING,
    MAPPING,
    MAPPING_HASH,
    STATS,
    nameloop2col,
    DICT_COLUMNS,
    RESERVED_COLUMNS,
    TENSORS,
    IN_PROGRESS_STATS,
    TAGS,
)
from pytimeloop.fastfusion.sim import TensorStorage, Tiling, Loop

from pytimeloop.fastfusion.util import fzs
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


# DEBUG_VISUALIZATION = Metrics.ALL_TENSORS | METRICS.PARTIAL_STATS

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
    einsum_id_to_name,
    rank_id_to_name,
    tensor_id_to_name,
    rank_name_to_shared_name,
    input_tensors: set[str],
    output_tensors: set[str],
    logfunc=None,
    metrics=Metrics.all_metrics(),
    tag_with: tuple[callable] = (),
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
    backing_storages = []
    all_storages = []
    intermediates_to_find = set(intermediate_tensors)
    found_tensors = set()
    ranks_remaining = {k: v for k, v in einsum_shape.items()}
    einsum_id = einsum_id_to_name[einsum_id]

    def record_storage(node):
        for dspace in node["dspace"]:
            storage = TensorStorage(
                tensor_id_to_name[dspace],
                len(full_tiling),
                node["target"],
                result.occupancy[(node["target"], dspace)],
            )
            all_storages.append(storage)
            if storage.tensor_id in intermediates_to_find:
                intermediates_to_find.remove(storage.tensor_id)
            if storage.tensor_id not in found_tensors:
                found_tensors.add(storage.tensor_id)
                backing_storages.append(storage)

        logstring.append(f"Strg({node['dspace']} in {node['target']})")

    def record_loop(node):
        nonlocal cur_idx
        if "tile_shape" in node:
            tile_shape = node["tile_shape"]
        else:
            tile_shape = shape[cur_idx]
            cur_idx += 1
        rank_id = equiv_groups.rank_to_group_id[node["rank"]]
        # rank_id = node["rank"]
        loop = Loop(
            rank_id_to_name[rank_id],
            tile_shape,
            node["type"] == "spatial",
        )
        ranks_remaining[node["rank"]] = tile_shape
        full_tiling.append(loop)
        logstring.append(f"{node['type'][0].upper()}{node['rank']} size {tile_shape}")

    logstring = []
    full_tiling = []
    for node in mapping:
        if node["type"] == "storage":
            record_storage(node)
        elif node["type"] == "spatial" or node["type"] == "temporal":
            record_loop(node)

    n_fused_loops = max(t.above_loop_index for t in backing_storages)
    tiling_full = Tiling(
        loops=tuple(full_tiling),
        tensors=frozenset(all_storages),
    )
    
    tagger_args = dict(
        einsum_id=einsum_id,
        backing_storages=backing_storages,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        tiling=tiling_full,
        rank_name_to_shared_name=rank_name_to_shared_name,
    )

    tiling_compatibility = Tiling(
        loops=tuple(full_tiling[:n_fused_loops]),
        tensors=frozenset(backing_storages),
        tags=fzs().union(*([set()] + [set(t(**tagger_args)) for t in tag_with]))
    )

    results = {}

    if Metrics.LATENCY in metrics:
        results["Latency"] = result.temporal_steps[einsum_id]

    if Metrics.ENERGY in metrics:
        results["Energy"] = energy

    offchip_ac = 0
    for (level, tensor, einsum), count in accesses.items():
        if level == 0:
            offchip_ac += count
        logstring.append(f"Ac_{level}_{tensor}={count:.2e}")

    if Metrics.OFF_CHIP_ACCESSES in metrics:
        results["Offchip Accesses"] = offchip_ac

    logstring.append(f"{result.fanout}")

    # Only record non-backing reservations. We'll reserve backing storage later
    # when we free the tensors & we know all operations for which the tensor must
    # be backed
    for r in all_storages:
        r: TensorStorage
        if r not in backing_storages:
            key = nameloop2col(r.backer_id, r.above_loop_index)
            results.setdefault(key, 0)
            results[key] += r.tile_size
        # logstring.append(f"{r}")

    if Metrics.LATENCY in metrics:
        logstring.append(f"L={results['Latency']:.2e}")

    if Metrics.ENERGY in metrics:
        logstring.append(f"E={results['Energy']:.2e}")

    if Metrics.OP_INTENSITY in metrics:
        results["Op_Intensity"] = result.op_intensity[1]

    logstring.append(f"Results: {results}")
    results[LOGSTRING] = {einsum_id: str(logstring)}
    results[MAPPING] = {einsum_id: tiling_full}
    results[TENSORS] = {einsum_id: backing_storages}
    results[STATS] = {
        einsum_id: {k: v for k, v in results.items() if k not in RESERVED_COLUMNS}
    }
    results[IN_PROGRESS_STATS] = {einsum_id: {}}
    results[MAPPING_HASH] = {einsum_id: hash((einsum_id, tiling_compatibility))}
    results[TAGS] = {einsum_id: tiling_compatibility.tags}

    key = (tiling_compatibility, fzs(results.keys()))

    is_pareto = True
    for prev_stats in compatibility_to_df[key]:
        keys = [k for k in results if k not in DICT_COLUMNS]
        if (
            fzs(prev_stats.keys()) == fzs(results.keys())
            and all(prev_stats[k] <= results[k] for k in keys)
            and any(prev_stats[k] < results[k] for k in keys)
        ):
            is_pareto = False
            break
    # TO DO: Index into the DF with both tiling compatibility and
    # the result keys
    if is_pareto:
        compatibility_to_df[key].append(results)
    results_return = {k: v for k, v in results.items() if k != LOGSTRING}
    return is_pareto, results_return, logstring
