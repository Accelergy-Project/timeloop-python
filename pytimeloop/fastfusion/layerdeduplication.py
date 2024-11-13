from collections import defaultdict
from itertools import permutations, product

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

    intermediate_tensors = get_intermediate_tensors(workload)

    all_tensor_properties = []
    all_tensors = [
        (einsum1_input_tensors, einsum1_output_tensor),
        (einsum2_input_tensors, einsum2_output_tensor)
    ]
    for input_tensors, output_tensors in all_tensors:
        tensor_properties = defaultdict(set)
        for tensor in input_tensors:
            tensor_properties[tensor].add('input')
        for tensor in output_tensors:
            tensor_properties[tensor].add('output')
        for tensor in tensor_properties:
            if tensor in intermediate_tensors:
                tensor_properties[tensor].add('intermediate')
        tensor_properties = {
            tensor: frozenset(properties)
            for tensor, properties in tensor_properties.items()
        }
        all_tensor_properties.append(tensor_properties)

    property_to_tensors = defaultdict(lambda: (set(), set()))
    for i, tensor_properties in enumerate(all_tensor_properties):
        for tensor, property in tensor_properties.items():
            tensor_sets = property_to_tensors[property]
            tensor_sets[i].add(tensor)

    # Check if we can rename tensors in einsum1 to einsum2
    for tensor_renaming in tensor_renamings(property_to_tensors):
        # Check if we can rename einsum1 ranks to create einsum2
        for renamed_ranks in permutations(einsum2_ranks):
            rank_renaming = {
                r1: r2 for r1, r2 in zip(einsum1_ranks, renamed_ranks)
            }
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


def tensor_renamings(property_to_tensors):
    for tensors_of_1, tensors_of_2 in property_to_tensors.values():
        if len(tensors_of_1) != len(tensors_of_2):
            return

    all_tensors_of_1 = [
        t
        for tensors_of_1, _ in property_to_tensors.values()
        for t in tensors_of_1
    ]
    permutations_of_tensor_2_by_property = []
    for _, tensors_of_2 in property_to_tensors.values():
        permutations_of_tensor_2_by_property.append(permutations(tensors_of_2))
    for permutation_of_2 in product(*permutations_of_tensor_2_by_property):
        permutation_of_2 = tuple(t for tupl in permutation_of_2 for t in tupl)
        renaming = dict(zip(all_tensors_of_1, permutation_of_2))
        yield renaming


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