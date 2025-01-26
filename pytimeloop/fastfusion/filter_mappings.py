from collections import defaultdict
from pytimeloop.fastfusion.sim import TensorStorage, Tiling
        
def get_ffmt_tag_mha(
        einsum_id: str, 
        backing_storages: set[TensorStorage], 
        input_tensors: set[str],
        output_tensors: set[str],
        tiling: Tiling,
        rank_name_to_shared_name: dict[str, str]
    ):
    B, H, M, F, P, G, E, D = (x + einsum_id for x in "BHMFPGED")
    
    einsum_id_to_input_output = {
        "Q": ["I_I_to_Q_K_V", "Q_Q_to_QK"],
        "K": ["I_I_to_Q_K_V", None], # NOTE: TANNER ADDED THESE
        "V": ["I_I_to_Q_K_V", None], # NOTE: TANNER ADDED THESE
        "QK": ["Q_Q_to_QK", "QK_QK_to_AV"],   # NOTE: K IS MISSING
        "AV": ["QK_QK_to_AV", "AV_AV_to_Z"], # NOTE: AV IS MISSING
        "Z": ["AV_AV_to_Z", "Z_Z_to_n"],
    }
    if einsum_id not in einsum_id_to_input_output:
        return ("FFMT_VALID",)
    a, b = einsum_id_to_input_output[einsum_id]
    
    tags = []
    
    min_weight_index = None
    max_non_weight_index = 0
    first, last = True, True
    for t in tiling.tensors:
        if t.backer_id != 1:
            continue
        if t.tensor_id in input_tensors and t in backing_storages:
            first = False
        if t.tensor_id in output_tensors and t in backing_storages:
            last = False
        # if "W_n_to_" in t.tensor_id:
        if t.tensor_id != a and t.tensor_id != b: # Weights!
            if min_weight_index is None:
                min_weight_index = t.above_loop_index
            else:
                min_weight_index = min(min_weight_index, t.above_loop_index)
        else:
            max_non_weight_index = max(max_non_weight_index, t.above_loop_index)
 
    if min_weight_index == 2:
        tags.append("FFMT_WEIGHTS_UNTILED")
    elif min_weight_index is None or min_weight_index < max_non_weight_index:
        tags.append("FFMT_WEIGHTS_INVALID")
    else:
        tags.append("FFMT_WEIGHTS_TILED")

    to_try = [([B, H], (2, 2)), ([B, H, M], (3, 3))]
    other_ranks = {
        "Q": [B, H, M, E, D],
        "K": [B, H, M, E, D],
        "V": [B, H, M, E, D],
        "QK": [B, H, M, P, E],
        "AV": [B, H, M, F, P],
        "Z": [B, H, M, G],
    }[einsum_id]

    valid = False
    if first and last: # Unfused
        to_try = []
        valid = True
        tags.append("FFMT_UNFUSED")
    elif first: # First Einsum in a chain
        to_try += [(other_ranks[:4], (3, 4)), (other_ranks, (5, 4))]
        tags.append("FFMT_FIRST")
    elif last: # Last Einsum in a chain
        to_try += [(other_ranks[4:], (3, 4))]
        tags.append("FFMT_LAST")
    else: # Middle Einsum in a chain
        if einsum_id == "AV":
            a, b = b, a
            other_ranks[-2], other_ranks[-1] = other_ranks[-1], other_ranks[-2]
        to_try += [(other_ranks[:4], (3, 4))]
        tags.append("FFMT_MIDDLE")

    for i, (c, (a_loops, b_loops)) in enumerate(to_try):
        perm = [rank_name_to_shared_name[x] for x in c] + ["*"]
        check_tensors = [TensorStorage(a, a_loops, 1, "*")]
        if b is not None:
            check_tensors.append(TensorStorage(b, b_loops, 1, "*"))
        if tiling.matches_permutation(perm):
            valid = valid
        if tiling.matches_permutation(perm) and tiling.has_tensor(*check_tensors):
            valid = True
            # tags.append(f"FFMT_VALID_{i}")

    # return ("FFMT_VALID" if valid else "FFMT_INVALID", weight_tag)
    if valid:# and weight_tag != "INVALID":
        return ("FFMT_VALID", *tags)
    return ("FFMT_INVALID",)
    
def get_tileflow_tag_mha(
        einsum_id: str, 
        backing_storages: set[TensorStorage], 
        input_tensors: set[str],
        output_tensors: set[str],
        tiling: Tiling,
        rank_name_to_shared_name: dict[str, str]
    ):
    # Valid iff it's an even mapping
    storage2level = defaultdict(set)
    for ts in backing_storages:
        storage2level[ts.backer_id].add(ts.above_loop_index)
    if all(len(s) == 1 for s in storage2level.values()):
        return ("TILEFLOW_VALID",)
    return ("TILEFLOW_INVALID",)
