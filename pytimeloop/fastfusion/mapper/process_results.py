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
    VALID,
)
from pytimeloop.fastfusion.sim import Tags, TensorStorage, Tiling, Loop

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
    DEBUG = auto()
    VALID = auto()

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
    tensor_to_relevant_ranks,
    logfunc=None,
    metrics=Metrics.all_metrics(),
    tag_with: tuple[callable] = (),
    copy_einsums: set[str] = (),
    prune=True,
    valid=True,
):
    if not prune:
        metrics = metrics | Metrics.VALID
    
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
    einsum_name = einsum_id_to_name[einsum_id]

    def record_storage(node):
        for dspace in node["dspace"]:
            storage = TensorStorage(
                tensor_id_to_name[dspace],
                len(full_tiling),
                node["target"],
                int(result.occupancy[(node["target"], dspace)]),
            )
            all_storages.append(storage)
            if storage.tensor_name in intermediates_to_find:
                intermediates_to_find.remove(storage.tensor_name)
            if storage.tensor_name not in found_tensors:
                found_tensors.add(storage.tensor_name)
                backing_storages.append(storage)
                
        if Metrics.DEBUG in metrics:
            logstring.append(f"Strg({node['dspace']} in {node['target']})")

    def record_loop(node):
        nonlocal cur_idx
        if "tile_shape" in node:
            tile_shape = node["tile_shape"]
        else:
            tile_shape = shape[cur_idx]
            cur_idx += 1
        # rank_name = equiv_groups.rank_to_group_id[node["rank"]]
        # rank_name = rank_name_to_shared_name[rank_id_to_name[node["rank"]]]
        rank_name = rank_id_to_name[node["rank"]]
        loop = Loop(
            fzs((str(rank_name),)), #rank_id_to_name[rank_name],
            tile_shape,
            node["type"] == "spatial",
        )
        ranks_remaining[node["rank"]] = tile_shape
        full_tiling.append(loop)
        if Metrics.DEBUG in metrics:
            logstring.append(f"{node['type'][0].upper()}{node['rank']} size {tile_shape}")

    logstring = []
    full_tiling = []
    for node in mapping:
        if node["type"] == "storage":
            record_storage(node)
        elif node["type"] == "spatial" or node["type"] == "temporal":
            record_loop(node)
        
    # If this Einsum is a copy op, consider only the movement from one backing
    # storage to another.
    copy_einsum = einsum_name in copy_einsums
    if copy_einsum:
        all_storages = backing_storages
        
    # If this Einsum is a copy op and the source and destination locations are
    # the same, then it is a null operation. Assume that the input tensors are
    # the output tensors. We don't want to double count, so get rid of the input
    # tensor occupancies. Note that we would like to keep the output tensor
    # occupancies in case their reservatinos get propagated to later Einsums. 
    null_copy_einsum = copy_einsum and len(set(t.storage_name for t in backing_storages)) == 1
    if null_copy_einsum:
        for i, r in enumerate(backing_storages):
            if r.tensor_name in input_tensors:
                backing_storages[i] = TensorStorage(
                    tensor_name=r.tensor_name,
                    above_loop_index=r.above_loop_index,
                    storage_name=r.storage_name,
                    tile_size=0,
                )


    n_fused_loops = max(t.above_loop_index for t in backing_storages)
    tiling_full = Tiling(
        loops=tuple(full_tiling),
        tensors=fzs(all_storages),
    )
    
    for i, l in enumerate(tiling_full.loops):
        for l2 in tiling_full.loops[i+1:]:
            if l.rank_name == l2.rank_name:
                assert l.bound >= l2.bound, f"{l} {l2}"
    
    tagger_args = dict(
        einsum_name=einsum_name,
        backing_storages=backing_storages,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        tiling=tiling_full,
        rank_name_to_shared_name=rank_name_to_shared_name,
        tensor_to_relevant_ranks=tensor_to_relevant_ranks,
    )
    
    tags = []
    for t in tag_with:
        tag = t(**tagger_args)
        assert isinstance(tag, tuple), "Tagger must return a tuple"
        tags.extend(tag)

    tiling_compatibility = Tiling(
        loops=tuple(full_tiling[:n_fused_loops]),
        tensors=fzs(backing_storages),
        tags=Tags(fzs(tags)),
    )

    results = {}

    if Metrics.LATENCY in metrics:
        results["Latency"] = result.temporal_steps[einsum_id]

    if Metrics.ENERGY in metrics:
        results["Energy"] = energy

    if Metrics.OFF_CHIP_ACCESSES in metrics:
        offchip_ac = 0
        for (level, tensor, einsum), count in accesses.items():
            if level == 0:
                offchip_ac += count
        results["Offchip Accesses"] = offchip_ac
        if Metrics.DEBUG in metrics:
            logstring.append(f"Ac_{level}_{tensor}={count:.2e}")
        
    if Metrics.DEBUG in metrics:
        logstring.append(f"{result.fanout}")

    # Only record non-backing reservations. We'll reserve backing storage later
    # when we free the tensors & we know all operations for which the tensor must
    # be backed.
    if not copy_einsum:
        for r in all_storages:
            r: TensorStorage
            if r not in backing_storages:
                key = nameloop2col(r.storage_name, min(r.above_loop_index, n_fused_loops))
                results.setdefault(key, 0)
                results[key] += r.tile_size

    if Metrics.LATENCY in metrics and Metrics.DEBUG in metrics:
        logstring.append(f"L={results['Latency']:.2e}")

    if Metrics.ENERGY in metrics and Metrics.DEBUG in metrics:
        logstring.append(f"E={results['Energy']:.2e}")

    if Metrics.OP_INTENSITY in metrics:
        results["Op_Intensity"] = result.op_intensity[1]
        
    if Metrics.VALID in metrics:
        results[VALID] = valid
        
    if metrics.DEBUG in metrics:
        logstring.append(f"Results: {results}")
        results[LOGSTRING] = {einsum_name: str(logstring)}
        results[STATS] = {einsum_name: {k: v for k, v in results.items() if k not in RESERVED_COLUMNS}}
        results[TAGS] = {einsum_name: tiling_compatibility.tags}
        results[MAPPING_HASH] = {einsum_name: hash((einsum_id, tiling_compatibility))}
        results[IN_PROGRESS_STATS] = {einsum_name: {k: v for k, v in results.items() if k not in RESERVED_COLUMNS}}
        results[TENSORS] = {einsum_name: backing_storages}

    results[MAPPING] = {einsum_name: tiling_full}

    if einsum_name in copy_einsums:
        if null_copy_einsum:
            for k, v in list(results.items()):
                results[k] = {} if k in DICT_COLUMNS else 0

    
    key = (tiling_compatibility, fzs(results.keys()))

    is_pareto = True
    if prune:
        for prev_stats in compatibility_to_df[key]:
            keys = [k for k in results if k not in DICT_COLUMNS]
            if (
                fzs(prev_stats.keys()) == fzs(results.keys())
                and all(prev_stats[k] <= results[k] for k in keys)
                and any(prev_stats[k] < results[k] for k in keys)
            ):
                is_pareto = False
                break
    if is_pareto:
        compatibility_to_df[key].append(results)
    results_return = {k: v for k, v in results.items() if k != LOGSTRING}
    # print(f"Results: {tiling_full.as_sorted_loopnest()}")
    return is_pareto, results_return, logstring
