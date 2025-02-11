from collections import defaultdict
from pytimeloop.fastfusion.sim import TensorStorage, Tiling


def get_looptree_tag_mha(
    einsum_name: str,
    backing_storages: set[TensorStorage],
    input_tensors: set[str],
    output_tensors: set[str],
    tiling: Tiling,
    rank_name_to_shared_name: dict[str, str],
    tensor_to_relevant_ranks,
):
    
    is_fused = any(t.storage_name != 0 for t in backing_storages)

    loops_above_backing_storages = max([0] + [
        t.above_loop_index for t in backing_storages if t.storage_name != 0
    ]
    )
    shared_loops = ",".join(l.rank_name for l in tiling.loops[:loops_above_backing_storages])
    if not is_fused:
        return ("LOOPTREE_VALID",)
    
    return ("LOOPTREE_VALID", "FUSED_LOOPS=" + shared_loops)
