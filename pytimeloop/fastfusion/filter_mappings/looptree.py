from collections import defaultdict
from pytimeloop.fastfusion.sim import TensorStorage, Tiling


def get_looptree_tag_mha(
    einsum_name: str,
    backing_storage: set[TensorStorage],
    input_tensors: set[str],
    output_tensors: set[str],
    tiling: Tiling,
    rank_name_to_shared_name: dict[str, str],   
    tensor_to_relevant_ranks,
):
    fused_storage = [t for t in backing_storage if t.memory_name != 0]
    
    # Unfused
    if not fused_storage:
        return ("LOOPTREE_VALID",)
    
    # Fused with one side but not the other. We don't want to interfere with the
    # unfused side, so just go LOOPTREE_VALID. The number of loops will be enforced
    # by the tiling since it must match for the one fused tensor.
    if len(fused_storage) == 1:
        return ("LOOPTREE_VALID",)
    
    # Fused with both sides. Make sure that the number of loops is the same.
    n_loops = set(t.above_loop_index for t in fused_storage)
    if len(n_loops) > 1:
        return ("LOOPTREE_INVALID",)
    return ("LOOPTREE_VALID", f"FUSED_LOOPS={n_loops.pop()}")
