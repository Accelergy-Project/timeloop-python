from pytimeloop.isl.singular import get_value_from_singular_qpolynomial
from pytimeloop.looptree.latency.processors import LATENCY_PROCESSORS

from bindings.looptree import SpatialTag


def compute_latency(mapping, temporal_steps, workload):
    return get_value_from_singular_qpolynomial(
        _compute_latency(mapping, 0, temporal_steps, workload)[1]
    ).to_python()


def _compute_latency(mapping, top_idx: int, temporal_steps, workload):
    einsum_name_to_id = workload.einsum_name_to_id()

    next_top_idx = top_idx
    for node in mapping:
        next_top_idx += 1

        if node['type'] in LATENCY_PROCESSORS.keys():
            children_latencies = [
                _compute_latency(branch, next_top_idx, temporal_steps, workload)
                for branch in node['branches']
            ]

            return LATENCY_PROCESSORS[node['type']](top_idx,
                                                    children_latencies)
        elif node['type'] == 'compute':
            einsum = node['einsum']
            if 'incomplete' in node and node['incomplete']:
                return ([], 0)
            einsum_id = einsum_name_to_id[einsum]
            return temporal_steps[einsum_id]


def ops_to_latency(dims, map):
    mask = [False]*len(dims)
    new_dims = []
    for i, d in enumerate(dims):
        if d == SpatialTag:
            mask[i] = True
        else:
            new_dims.append(d)
    return map.domain().identity().card()