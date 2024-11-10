from collections import defaultdict
from itertools import permutations

from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors


def group_similar_einsums(workload, analyzer):
    ref_to_others = {}



def is_equivalent(einsum_id1, einsum_id2, workload, analyzer):
    einsum1_ranks = workload.einsum_ospace_dimensions(einsum_id1)
    einsum2_ranks = workload.einsum_ospace_dimensions(einsum_id2)

    if len(einsum1_ranks) != len(einsum2_ranks):
        return None, None

    einsum1_input_tensors = workload.tensors_read_by_einsum(einsum_id1)
    einsum1_output_tensor = workload.tensors_written_by_einsum(einsum_id1)
    einsum2_input_tensors = workload.tensors_read_by_einsum(einsum_id2)
    einsum2_output_tensor = workload.tensors_written_by_einsum(einsum_id2)

    if einsum1_output_tensor is None:
        einsum1_output_tensor = set()
    if einsum2_output_tensor is None:
        einsum2_output_tensor = set()

    einsum1_tensors = einsum1_input_tensors | einsum1_output_tensor
    einsum2_tensors = einsum2_input_tensors | einsum2_output_tensor

    intermediate_tensors = get_intermediate_tensors(workload)

    tensor_properties = defaultdict(set)
    for tensor in einsum1_input_tensors | einsum2_input_tensors:
        tensor_properties[tensor].add('input')
    for tensor in einsum1_output_tensor | einsum2_output_tensor:
        tensor_properties[tensor].add('input')
    for tensor in intermediate_tensors:
        if tensor not in tensor_properties:
            continue
        tensor_properties[tensor].add('intermediate')
    tensor_properties = {
        tensor: frozenset(properties)
        for tensor, properties in tensor_properties.items()
    }
    property_to_tensors = defaultdict(lambda: (set(), set()))
    for tensor, property in tensor_properties:
        tensor_sets = property_to_tensors[property]
        if tensor in einsum1_tensors:
            tensor_sets[0].add(tensor)
        else:
            tensor_sets[1].add(tensor)

    for tensor_sets in property_to_tensors.values():
        if len(tensor_sets[0]) != len(tensor_sets[1]):
            return None, None



    # Check if we can rename einsum1 ranks to create einsum2
    for renamed_ranks in permutations(einsum2_ranks):
        rank_renaming = {
            r1: r2 for r1, r2 in zip(einsum1_ranks, renamed_ranks)
        }
        # for tensor_renaming in get_tensor_renamings(property_to_tensors):
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