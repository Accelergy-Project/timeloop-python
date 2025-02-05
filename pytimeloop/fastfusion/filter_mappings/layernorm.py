from pytimeloop.fastfusion.sim import TensorStorage, Tiling


LAYERNORM_VALID = "LAYERNORM_VALID"
LAYERNORM_INVALID = "LAYERNORM_INVALID"


def get_layernorm_tag_mha(
        einsum_name: str, 
        backing_storages: set[TensorStorage], 
        input_tensors: set[str],
        output_tensors: set[str],
        tiling: Tiling,
        rank_name_to_shared_name: dict[str, str],
        tensor_to_relevant_ranks,
    ):
    EINSUMS_ADJACENT_TO_LAYERNORM = {"Z", "FFA"}
    if einsum_name not in EINSUMS_ADJACENT_TO_LAYERNORM:
        return (LAYERNORM_VALID,)

    normalized_tensor = "Z_Z_to_FFA"

    for t in backing_storages:
        if t.tensor_id != normalized_tensor:
            continue
        if t.backer_id != 1:
            continue

        normalized_tensor_idx = t.above_loop_index
        for l in tiling.loops[:normalized_tensor_idx]:
            if l.rank_id == "GZ":
                return (LAYERNORM_INVALID,)

    return (LAYERNORM_VALID,)