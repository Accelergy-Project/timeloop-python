from collections import defaultdict
from pytimeloop.fastfusion.sim import TensorStorage, Tiling

FFMT_VALID = "FFMT_VALID"
FFMT_INVALID = "FFMT_INVALID"
FFMT_WEIGHT_UNTILED = "FFMT_WEIGHT_UNTILED"
FFMT_WEIGHT_TILED = "FFMT_WEIGHT_TILED"

def is_even(tiling: Tiling, tensor_to_relevant_ranks, skip_tensors=None):
    if skip_tensors is None:
        skip_tensors = set()
    # Highest index of all storage nodes of a given hardware level.
    # This is the "canonical" even storage node, which is where storage
    # nodes for even exploration is nominally placed without LRP.
    storage2highestindex = defaultdict(lambda: 0)
    for ts in tiling.storage:
        if any(t in ts.tensor_name for t in skip_tensors):
            continue
        storage2highestindex[ts.memory_name] = max(
            storage2highestindex[ts.memory_name], ts.above_loop_index
        )

    for ts in tiling.storage:
        if any(t in ts.tensor_name for t in skip_tensors):
            continue
        highest_idx = storage2highestindex[ts.memory_name]
        lowest_idx = ts.above_loop_index
        # If any relevant rank separates a tensor's storage node from the
        # "canonical" even storage node, then the tiling is uneven
        if any(
            l.rank_name in tensor_to_relevant_ranks[ts.tensor_name]
            for l in tiling.loops[lowest_idx:highest_idx]
        ):
            return False
    return True

OPTIMUS_INVALID = "OTMS_INVALID"
OPTIMUS_VALID = "OPTIMUS_VALID"

def get_optimus_tag(
        einsum_name: str, 
        backing_storage: set[TensorStorage], 
        input_tensors: set[str],
        output_tensors: set[str],
        tiling: Tiling,
        rank_name_to_shared_name: dict[str, str],
        tensor_to_relevant_ranks,
    ):
    # Tag includes:
    # - Number of fused loops. Must be equal for all fused Einsums
    # - All fused tensors must be backed under the same number of loops
    # - Weights stored at the same loop index for all fused operations.
    #   They are also all stored:
    #   - Above all fused loops
    #   - Below all fused loops. May be further below if there are relevant
    #   loops below
    intermediates = input_tensors | output_tensors
    fused_storage = [t for t in backing_storage if t.memory_name != 0]

    # All fused tensors must be backed under the same number of loops
    n_fused_loops = set(t.above_loop_index for t in fused_storage)
    if len(n_fused_loops) != 1:
        return (OPTIMUS_INVALID,)
    n_fused_loops = n_fused_loops.pop()
    
    # Unfused is vaild
    if n_fused_loops == 0:
        return (OPTIMUS_VALID,)
    
    # Weights must be stored either above all or below all fused loops
    weight_first_non_backing_storages = []
    for t in tiling.storage:
        if t.tensor_name not in intermediates and t not in backing_storage:
            weight_first_non_backing_storages.append(t)
    weight_non_backing_above_index = set(t.above_loop_index for t in weight_first_non_backing_storages)
    weights_above = all(w == 0 for w in weight_non_backing_above_index)
    weights_below = all(w >= n_fused_loops for w in weight_non_backing_above_index)
    
    if not (weights_above or weights_below):
        return (OPTIMUS_INVALID,)
    
    return (
        OPTIMUS_VALID,
        f"OPTIMUS_N_FUSED_LOOPS={n_fused_loops}",
        f"OPTIMUS_WEIGHTS_ABOVE={weights_above}",
    )
    
    # Fused groups of tensors are executed sequentially Within a fused group,
    # you fuse everything While we're exploring SIMs, look at live tensors. They
    # should either be all fused or all not fused. We can't have both.
    
    # Our Optimus implementation:
    # - Assume that they have Pareto pruning at every step. This is equivalent
    #   to their dynamic programming. We can generously extend their dynamic
    #   programming to more dataflows by assuming that their dynamic programming
    #   would save, for each Einsum, the mapping for each possible inter-layer
    #   mapping. This is generous because we are giving them our compatibility
    #   comparison metrics.
    # 
    # - They can not lookeahead filter
    # - They iterate through every pair of mappings for every Einsum to check
    #   for compatibility, rather than hashing comparison metrics.
    # - They don't have lifetime metrics. This means:
    #   - When they put Einsum(s) together, they must re-evaluate the memory of
    #     the partial mapping. Therefore, for the Nth Einsum, the evaluation
    #     cost is N
    #   - If two partial mappings have any reservations, they can't be compared.
    # - No same-shape layer optimization
    
    # We're giving them our intra-layer mapper ability to stop increasing tile
    # size when buffer capacity is exceeded. This is generous because they give
    # no information on the intra-layer mapper.

    # Q for Michael: I think we should have the same-shape optimization on for
    # us for GPT but not matmuls. Matmuls feels unfair because we set it up with
    # many repeated shapes.
    
    B, H, M, F, P, G, E, D, C, J = (x + einsum_name for x in "BHMFPGEDCJ")
    EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK = {
        "Q":   [D, E],
        "K":   [D, E],
        "V":   [D, F],
        "QK":  [E, P],
        "AV":  [P, F],
        "Z":   [F, G],
        "FFA": [G, C],
        "FFB": [C, J]
    }
    
    unfused = all(t.memory_name == 0 for t in backing_storage)
    if "Matmul" in einsum_name:
        min_weight_idx, max_weight_idx, max_non_weight_idx = float('inf'), 0, 0
        max_weight_idx = 0
        if unfused:
            return (FFMT_VALID,)
        untiled_fused = all(t.above_loop_index == 0 for t in backing_storage)
        if untiled_fused:
            return (FFMT_VALID, )

        for t in tiling.storage:
            is_weight = "Filter" in t.tensor_name
            if is_weight:
                min_weight_idx = min(min_weight_idx, t.above_loop_index)
                max_weight_idx = max(max_weight_idx, t.above_loop_index)
            else:
                max_non_weight_idx = max(max_non_weight_idx, t.above_loop_index)
        
        weight_untiled = (
            min_weight_idx == 0
            and
            max_weight_idx == 0
        )
        if weight_untiled:
            return (FFMT_VALID, FFMT_WEIGHT_UNTILED)
        elif min_weight_idx >= max_non_weight_idx:
            return (FFMT_VALID, FFMT_WEIGHT_TILED)
        return (FFMT_INVALID,)

    if einsum_name not in EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK:
        if unfused:
            return (FFMT_VALID,)
        return (FFMT_INVALID,)

    reduced_rank, output_rank = EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK[einsum_name]

    EINSUM_NAME_TO_INPUT_OUTPUT_TENSORS = {
        "Q":   ["I_I_to_Q_K_V",   "Q_Q_to_QK"],
        "K":   ["I_I_to_Q_K_V",   "K_K_to_QK"],
        "V":   ["I_I_to_Q_K_V",   "V_V_to_AV"],
        "QK":  ["Q_Q_to_QK",      "QK_QK_to_AV"],
        "AV":  ["QK_QK_to_AV",    "AV_AV_to_Z"],
        "Z":   ["AV_AV_to_Z",     "Z_Z_to_FFA"],
        "FFA": ["Z_Z_to_FFA",     "FFA_FFA_to_FFB"],
        "FFB": ["FFA_FFA_to_FFB", "FFB_FFB_to_n"]
    }

    input_tensor, output_tensor = EINSUM_NAME_TO_INPUT_OUTPUT_TENSORS[einsum_name]
    input_output_tensors = {input_tensor, output_tensor}

    min_weight_idx = float('inf')
    max_weight_idx = 0
    max_non_weight_idx = 0
    first, last = True, True
    for t in tiling.storage:
        if t.memory_name != 1:
            continue
        if t.tensor_name == input_tensor and t in backing_storage:
            first = False
        if t.tensor_name == output_tensor and t in backing_storage:
            last = False

        is_weight = t.tensor_name not in input_output_tensors
        if is_weight:
            min_weight_idx = min(min_weight_idx, t.above_loop_index)
            max_weight_idx = max(max_weight_idx, t.above_loop_index)
        else:
            max_non_weight_idx = max(max_non_weight_idx, t.above_loop_index)

    unfused = first and last
    if unfused:
        return (FFMT_VALID,)

    FFMT_CANNOT_FUSE = {"K", "V"}
    if einsum_name in FFMT_CANNOT_FUSE:
        return (FFMT_INVALID,)

    prefix_choices = [
        ([B, H], (2, 2))
    ]

    unfused = False
    extra_rank_choices = [
        ([M], (1, 1)),
    ]
    if first and last:
        unfused = True
    elif first:
        if output_rank is not None:
            extra_rank_choices.append((
                [M, output_rank],
                (1, 2)
            ))
        if reduced_rank is not None and output_rank is not None:
            extra_rank_choices.append((
                [M, output_rank, reduced_rank],
                (3, 2)
            ))
        if output_rank is None and reduced_rank is not None:
            extra_rank_choices.append((
                [M, reduced_rank],
                (2, 1)
            ))
    elif last:
        if output_rank is not None:
            extra_rank_choices.append((
                [M, output_rank],
                (1, 2)
            ))
    else:
        if reduced_rank is not None:
            extra_rank_choices.append((
                [M, reduced_rank],
                (2, 1)
            ))

    for prefix_permutation, prefix_storage in prefix_choices:
        for extra_permutation, extra_storage in extra_rank_choices:
            permutation = prefix_permutation + extra_permutation
            input_storage = prefix_storage[0] + extra_storage[0]
            output_storage = prefix_storage[1] + extra_storage[1]
            untiled_weight_idx = len(prefix_permutation)

            check_tensors = [
                TensorStorage(input_tensor, input_storage, 1, "*"),
                TensorStorage(output_tensor, output_storage, 1, "*")
            ]

            if not tiling.matches_permutation(permutation):
                continue
            if not tiling.has_tensor(*check_tensors):
                continue

            # INVARIANCE: at this point, loops[0] must be over batch
            # and loops[1] must be over heads
            if tiling.loops[0].bound != 1:   # TODO: `bound` should be `shape`
                continue
            if tiling.loops[1].bound != 1:
                continue

            weight_untiled = (
                min_weight_idx == untiled_weight_idx
                and
                max_weight_idx == untiled_weight_idx
            )
            if weight_untiled:
                return (FFMT_VALID, FFMT_WEIGHT_UNTILED)
            elif min_weight_idx >= max_non_weight_idx:
                return (FFMT_VALID, FFMT_WEIGHT_TILED)

    return (FFMT_INVALID,)
