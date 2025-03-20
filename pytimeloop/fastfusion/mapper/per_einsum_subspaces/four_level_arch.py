from .subspaces import (
    infer_smallest_tile_shape,
    LinearMapping,
    make_storage,
    make_temporal_fors,
    make_temporal_fors_with_smallest_tile,
    make_spatial_fors
)
from pytimeloop.looptree.mapping_utilities import get_last_fused_loop_idx
from pytimeloop.fastfusion.mapper.constraints import DataflowConstraint


"""
Mapping constraint:
DRAM
----
for loop over any rank
----
GlobalBuffer
----
par-for loop over H
par-for loop over M
for loop over any rank
----
LocalBuffer
----
par-for loop over M
par-for loop over C
for loop over any rank
----
Register
----
leftover for loop
"""

EINSUM_ID_TO_FULLY_PARALLEL_RANKS = {
    "I": {"BI", "MI", "DI"},
    "Q": set(),
    "K": set(),
    "V": set(),
    "QK": {"HQK"},
    "AV": {"HAV"},
    "Z": set(),
    "FFA": set(),
    "FFB": set(),
}

EINSUM_ID_TO_OUTPUT_PARALLEL_RANKS = {
    "I": set(),
    "Q": {"HQ", "EQ"},
    "K": {"HK", "EK"},
    "V": {"HV", "EV"},
    "QK": {"PQK"},
    "AV": {"FAV"},
    "Z": {"GZ"},
    "FFA": {"CFFA"},
    "FFB": {"CFFB"},
}

EINSUM_ID_TO_REDUCED_RANKS = {
    "I": set(),
    "Q": {"DQ"},
    "K": {"DK"},
    "V": {"DV"},
    "QK": {"EQK"},
    "AV": {"PAV"},
    "Z": {"HZ", "FZ"},
    "FFA": {"GFFA"},
    "FFB": {"JFFB"},
}

EINSUM_ID_TO_WEIGHT_LIKE_TENSOR = {
    "I": "I_n_to_I",
    "Q": "W_n_to_Q",
    "K": "W_n_to_K",
    "V": "W_n_to_V",
    "QK": "K_K_to_QK",
    "AV": "V_V_to_AV",
    "Z": "W_n_to_Z",
    "FFA": "W_n_to_FFA",
    "FFB": "W_n_to_FFB",
}

for i in range(1, 32):
    matmul_name = f"Matmul{i}"
    m, k, n = f"M{i}", f"K{i}", f"N{i}"
    EINSUM_ID_TO_FULLY_PARALLEL_RANKS[matmul_name] = set()
    EINSUM_ID_TO_OUTPUT_PARALLEL_RANKS[matmul_name] = {n}
    EINSUM_ID_TO_REDUCED_RANKS[matmul_name] = {k}
    EINSUM_ID_TO_WEIGHT_LIKE_TENSOR[matmul_name] = f"Filter{i}"

def make_subspaces(tensors,
                   intermediate_tensors,
                   tensor_to_relevant_ranks,
                   einsum_id,
                   workload):
    """
    fully_parallel_ranks: in all tensors
    output_parallel_ranks: not in input
    reduced_ranks: not in output
    weight_like_tensor: tensor that will be stationary in systolic array
    """
    einsum_name = workload.einsum_id_to_name()[einsum_id]
    fully_parallel_ranks = {
        workload.dimension_name_to_id()[r]
        for r in EINSUM_ID_TO_FULLY_PARALLEL_RANKS[einsum_name]
    }
    output_parallel_ranks = {
        workload.dimension_name_to_id()[r]
        for r in EINSUM_ID_TO_OUTPUT_PARALLEL_RANKS[einsum_name]
    }
    reduced_ranks = {
        workload.dimension_name_to_id()[r]
        for r in EINSUM_ID_TO_REDUCED_RANKS[einsum_name]
    }
    weight_like_tensor = EINSUM_ID_TO_WEIGHT_LIKE_TENSOR[einsum_name]
    weight_like_tensor = workload.data_space_name_to_id()[weight_like_tensor]
    output_tensor = next(iter(workload.tensors_written_by_einsum(einsum_id)))

    def off_chip_storage(mapping):
        off_chip_must_retain = tensors - intermediate_tensors
        off_chip_can_retain = intermediate_tensors
        yield from make_storage(
            mapping,
            level=0,
            must_retain_tensors=off_chip_must_retain,
            can_retain_tensors=off_chip_can_retain,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=False,
            add_split_at_tensors=intermediate_tensors,
            return_retained_tensors=True,
            apply_lrp_after_loop_idx=None,
        )

    all_ranks = list(sorted(workload.einsum_ospace_dimensions(einsum_id)))

    def fused_temporal_fors(mapping, unfused_tensors):
        for partial_mapping in make_temporal_fors(mapping,
                                                  all_ranks,
                                                  min_loops=0,
                                                  ):
            yield partial_mapping, unfused_tensors


    def glb_storage(mapping, unfused_tensors):
        glb_fused_tensors = intermediate_tensors - unfused_tensors
        last_fused_loop_idx = get_last_fused_loop_idx(mapping, intermediate_tensors)
        # last_fused_loop_idx = None
        for partial_mapping in make_storage(mapping,
                                            level=1,
                                            must_retain_tensors=intermediate_tensors,
                                            can_retain_tensors=set(),
                                            must_fully_reuse_tensors=glb_fused_tensors,
                                            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                            explore_uneven=True,
                                            add_split_at_tensors=glb_fused_tensors,
                                            must_have_terminal_storage=False,
                                            apply_lrp_after_loop_idx=last_fused_loop_idx):
            last_fused_loop_idx = get_last_fused_loop_idx(partial_mapping, intermediate_tensors)
            for pm2 in make_storage(partial_mapping,
                                    level=1,
                                    must_retain_tensors=set(),
                                    can_retain_tensors=tensors - intermediate_tensors,
                                    must_fully_reuse_tensors=set(),
                                    tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                    explore_uneven=True,
                                    add_split_at_tensors=set(),
                                    must_have_terminal_storage=True,
                                    apply_lrp_after_loop_idx=last_fused_loop_idx):

                success = True
                for i in range(last_fused_loop_idx + 1, len(pm2) - 1):
                    n1, n2 = pm2[i], pm2[i+1]
                    for ntype in ["temporal", "spatial"]:
                        if n1["type"] == ntype and n2["type"] == ntype:
                            if n1["rank"] <= n2["rank"]:
                                success = False
                                break
                if success:
                    yield pm2

    def core_spatial_fors(mapping):
        ranks = fully_parallel_ranks | output_parallel_ranks
        yield from make_spatial_fors(mapping, ranks, 4, unordered=True)

    input_output_ranks = \
        set(all_ranks) - output_parallel_ranks - reduced_ranks - fully_parallel_ranks
    def core_temporal_fors(mapping):
        for pm in make_temporal_fors_with_smallest_tile(mapping,
                                                        fully_parallel_ranks,
                                                        unordered=True):
            for pm2 in make_temporal_fors(pm, input_output_ranks,
                                          unordered=True):
                yield from make_temporal_fors(pm2,
                                              output_parallel_ranks,
                                              unordered=True)

    def llb_storage(mapping):
        yield from make_storage(mapping,
                                level=2,
                                must_retain_tensors={output_tensor},
                                can_retain_tensors=set(),
                                tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                explore_uneven=False)

    def pe_spatial_fors(mapping):
        for pm in make_spatial_fors(mapping, output_parallel_ranks, 128, unordered=True):
            yield from make_spatial_fors(pm, reduced_ranks, 128, unordered=True)

    def pe_temporal_fors(mapping):
        yield from make_temporal_fors_with_smallest_tile(mapping,
                                                         reduced_ranks,
                                                         unordered=True)

    def register_storage(mapping):
        yield from make_storage(mapping,
                                level=3,
                                must_retain_tensors={weight_like_tensor},
                                can_retain_tensors=set(),
                                tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                explore_uneven=False)

    def mac_temporal_fors(mapping):
        yield from make_temporal_fors_with_smallest_tile(mapping,
                                                         input_output_ranks,
                                                         unordered=True)

    def mac(mapping):
        mapping.add_compute(einsum_id, 2)
        yield mapping

    return [
        lambda: [LinearMapping()],
        off_chip_storage,
        fused_temporal_fors,
        glb_storage,
        core_spatial_fors,
        core_temporal_fors,
        llb_storage,
        pe_spatial_fors,
        pe_temporal_fors,
        register_storage,
        mac_temporal_fors,
        mac
    ]