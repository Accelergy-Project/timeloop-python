from pytimeloop.isl.singular import get_value_from_singular_qpolynomial

from pytimeloop.looptree.latency.processors import LATENCY_PROCESSORS


def compute_latency(mapping, temporal_steps, workload):
    latencies = {
        k: steps_to_latency(v) for k, v in temporal_steps.items()
    }
    return get_value_from_singular_qpolynomial(
        _compute_latency(mapping, 0, latencies, workload)
    )


def _compute_latency(mapping, top_idx: int, latencies, workload):
    einsum_name_to_id = workload.einsum_name_to_id()

    next_top_idx = top_idx
    for node in mapping:
        next_top_idx += 1

        if node['type'] in LATENCY_PROCESSORS.keys():
            children_latencies = [
                _compute_latency(branch, next_top_idx, latencies, workload)
                for branch in node['branches']
            ]

            return LATENCY_PROCESSORS[node['type']](top_idx,
                                                    children_latencies)
        elif node['type'] == 'compute':
            einsum = node['einsum']
            einsum_id = einsum_name_to_id[einsum]
            return latencies[einsum_id]


def steps_to_latency(map):
    return map.domain().identity().card()