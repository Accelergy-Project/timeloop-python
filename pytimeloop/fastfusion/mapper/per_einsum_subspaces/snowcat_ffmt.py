from pytimeloop.fastfusion.mapper.per_einsum_mapper import LinearMapping, make_storage, make_temporal_fors, make_temporal_fors_with_smallest_tile

def make_ffmt_subspaces(tensors,
                        intermediate_tensors,
                        tensor_to_relevant_ranks,
                        einsum_id,
                        workload):
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
        )

    # TODO: this is brittle
    all_ranks = list(sorted(workload.einsum_ospace_dimensions(einsum_id)))
    all_ranks = list(sorted(all_ranks))
    M = all_ranks[0]
    N = all_ranks[1]
    K = all_ranks[2]

    if einsum_id == 0:
        allowed_fused_ranks = all_ranks
    elif einsum_id == 1:
        allowed_fused_ranks = {M, K}
    elif einsum_id == max(workload.einsum_id_to_name().keys()):
        allowed_fused_ranks = {M, N}
    else:
        allowed_fused_ranks = {M}

    def fused_temporal_fors(mapping, unfused_tensors):
        for partial_mapping in make_temporal_fors(mapping, allowed_fused_ranks):
            # for partial_mapping in make_temporal_fors(mapping, all_ranks):
                for partial_mapping in make_temporal_fors_with_smallest_tile(mapping, all_ranks):
                    yield partial_mapping, unfused_tensors


    def glb_storage(mapping, unfused_tensors):
        glb_fused_tensors = intermediate_tensors - unfused_tensors
        yield from make_storage(
            mapping,
            level=1,
            must_retain_tensors=tensors,
            can_retain_tensors=set(),
            must_fully_reuse_tensors=glb_fused_tensors,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=True,
            add_split_at_tensors=glb_fused_tensors
        )

    def mac(mapping):
            mapping.add_compute(einsum_id, 2)
            yield mapping

    return [
        lambda: [LinearMapping()],
        off_chip_storage,
        fused_temporal_fors,
        glb_storage,
        mac
    ]