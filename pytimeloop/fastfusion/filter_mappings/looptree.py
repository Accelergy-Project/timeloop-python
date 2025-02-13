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
    if not is_fused:
        return ("LOOPTREE_VALID",)

    n_loops_above_backing_storages = max([0] + [
        t.above_loop_index for t in backing_storages if int(t.storage_name) != 0
    ]
    )
    
    return ("LOOPTREE_VALID", f"FUSED_LOOPS={n_loops_above_backing_storages}")
