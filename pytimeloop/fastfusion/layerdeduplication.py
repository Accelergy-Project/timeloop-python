from itertools import permutations


def is_equivalent(einsum_id1, einsum_id2, workload, analyzer):
    einsum1_ranks = workload.einsum_ospace_dimensions(einsum_id1)
    einsum2_ranks = workload.einsum_ospace_dimensions(einsum_id2)

    if len(einsum1_ranks) != len(einsum2_ranks):
        return None, None

    einsum1_input_tensors = workload.tensors_read_by_einsum(einsum_id1)
    einsum1_output_tensor = workload.tensors_written_by_einsum(einsum_id1)
    einsum2_input_tensors = workload.tensors_read_by_einsum(einsum_id2)
    einsum2_output_tensor = workload.tensors_written_by_einsum(einsum_id2)

    if einsum1_output_tensor is None and einsum2_output_tensor is not None:
        return None, None
    if einsum1_output_tensor is not None and einsum2_output_tensor is None:
        return None, None

    # Check if we can rename einsum1 ranks to create einsum2
    for renamed_ranks in permutations(einsum2_ranks):
        rank_renaming = {
            r1: r2 for r1, r2 in zip(einsum1_ranks, renamed_ranks)
        }
        for renamed_input_tensors in permutations(einsum2_input_tensors):
            input_tensor_renaming = {
                t1: t2 for t1, t2
                in zip(einsum1_input_tensors, renamed_input_tensors)
            }
            for renamed_output_tensors in permutations(einsum2_output_tensor):
                output_tensor_renaming = {
                    t1: t2 for t1, t2
                    in zip(einsum1_output_tensor, renamed_output_tensors)
                }
                tensor_renaming = input_tensor_renaming | output_tensor_renaming

                if not _shape_is_equivalent(rank_renaming, workload):
                    continue

                if not _dependency_is_equivalent(einsum_id1,
                                                einsum_id2,
                                                rank_renaming,
                                                tensor_renaming,
                                                analyzer):
                    continue

                return rank_renaming, tensor_renaming

    return None, None


def _shape_is_equivalent(rank_renaming, workload):
    for r1, r2 in rank_renaming.items():
        r1_shape = workload.get_rank_shape(r1)
        r2_shape = workload.get_rank_shape(r2)
        if r1_shape != r2_shape:
            return False
    return True


def _dependency_is_equivalent(einsum_id1,
                              einsum_id2,
                              rank_renaming,
                              tensor_renaming,
                              analyzer):
    for t1, t2 in tensor_renaming.items():
        for r1, r2 in rank_renaming.items():
            r1_relevant_to_t1 = \
                analyzer.einsum_dim_is_directly_relevant_to_tensor(
                    einsum_id1,
                    r1,
                    t1
                )
            r2_relevant_to_t2 = \
                analyzer.einsum_dim_is_directly_relevant_to_tensor(
                    einsum_id2,
                    r2,
                    t2
                )
            if r1_relevant_to_t1 != r2_relevant_to_t2:
                return False
    return True