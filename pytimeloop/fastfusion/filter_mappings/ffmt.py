from pytimeloop.fastfusion.sim import TensorStorage, Tiling

FFMT_VALID = "FFMT_VALID"
FFMT_INVALID = "FFMT_INVALID"
FFMT_WEIGHT_UNTILED = "FFMT_WEIGHT_UNTILED"
FFMT_WEIGHT_TILED = "FFMT_WEIGHT_TILED"


def get_ffmt_tag_mha(
        einsum_name: str, 
        backing_storages: set[TensorStorage], 
        input_tensors: set[str],
        output_tensors: set[str],
        tiling: Tiling,
        rank_name_to_shared_name: dict[str, str],
        tensor_to_relevant_ranks,
    ):
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
    for t in tiling.tensors:
        if t.backer_id != 1:
            continue
        if t.tensor_id == input_tensor and t in backing_storages:
            first = False
        if t.tensor_id == output_tensor and t in backing_storages:
            last = False

        is_weight = t.tensor_id not in input_output_tensors
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
        ([B], (1, 1)),
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

    for prefix_permutation, prefix_storages in prefix_choices:
        for extra_permutation, extra_storages in extra_rank_choices:
            permutation = prefix_permutation + extra_permutation
            input_storage = prefix_storages[0] + extra_storages[0]
            output_storage = prefix_storages[1] + extra_storages[1]
            untiled_weight_idx = len(prefix_permutation)

            check_tensors = [
                TensorStorage(input_tensor, input_storage, 1, "*"),
                TensorStorage(output_tensor, output_storage, 1, "*")
            ]

            if not tiling.matches_permutation(permutation):
                continue
            if not tiling.has_tensor(*check_tensors):
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
