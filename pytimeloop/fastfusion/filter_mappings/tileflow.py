from collections import defaultdict
from pytimeloop.fastfusion.sim import TensorStorage, Tiling


def get_tileflow_tag_mha(
    einsum_id: str,
    backing_storages: set[TensorStorage],
    input_tensors: set[str],
    output_tensors: set[str],
    tiling: Tiling,
    rank_name_to_shared_name: dict[str, str],
    tensor_to_relevant_ranks,
):
    
    is_fused = any(t.storage_name != 0 for t in backing_storages)

    # Valid iff it's an even mapping
    if not is_even(tiling, tensor_to_relevant_ranks):
        return ("TILEFLOW_INVALID",)
    
    loops_above_backing_storages = max([0] + [
        t.above_loop_index for t in backing_storages if t.storage_name != 0
    ]
        
    )
    shared_loops = ",".join(l.rank_name for l in tiling.loops[:loops_above_backing_storages])
    if not is_fused:
        return ("TILEFLOW_VALID",)
    
    return ("TILEFLOW_VALID", "FUSED_LOOPS=" + shared_loops)


def is_even(tiling: Tiling, tensor_to_relevant_ranks):
    # Highest index of all storage nodes of a given hardware level.
    # This is the "canonical" even storage node, which is where storage
    # nodes for even exploration is nominally placed without LRP.
    storage2highestidx = defaultdict(lambda: 0)
    for ts in tiling.tensors:
        storage2highestidx[ts.storage_name] = max(
            storage2highestidx[ts.storage_name], ts.above_loop_index
        )

    for ts in tiling.tensors:
        highest_idx = storage2highestidx[ts.storage_name]
        lowest_idx = ts.above_loop_index
        # If any relevant rank separates a tensor's storage node from the
        # "canonical" even storage node, then the tiling is uneven
        if any(
            l.rank_name in tensor_to_relevant_ranks[ts.tensor_name]
            for l in tiling.loops[lowest_idx:highest_idx]
        ):
            return False
    return True
