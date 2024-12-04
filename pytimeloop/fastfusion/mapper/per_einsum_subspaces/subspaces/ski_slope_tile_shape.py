from functools import partial, reduce
from operator import and_


def infer_smallest_tile_shape(mapping,
                              workload,
                              einsum_id,
                              tensor_to_relevant_ranks):
    tensors = workload.tensors_read_by_einsum(einsum_id)
    tensors |= workload.tensors_written_by_einsum(einsum_id)

    unstored_tensor = set(tensors)

    relevant_ranks = partial(
        _ranks_relevant_to_tensors,
        tensor_to_relevant_ranks=tensor_to_relevant_ranks
    )

    ranks_relevant_to_unstored = relevant_ranks(unstored_tensor)
    mapping = mapping.copy()
    for node in mapping:
        if node["type"] == "temporal":
            rank = node["rank"]
            if rank in ranks_relevant_to_unstored:
                node["tile_shape"] = 1
        elif node["type"] == "storage":
            tensors_at_node = set(node["dspace"])
            unstored_tensor -= tensors_at_node
            ranks_relevant_to_unstored = relevant_ranks(unstored_tensor)
    yield mapping


def _ranks_relevant_to_tensors(tensors, tensor_to_relevant_ranks):
    return reduce(
        and_,
        [tensor_to_relevant_ranks[t] for t in tensors],
        set()
    )

