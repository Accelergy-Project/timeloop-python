from pytimeloop.looptree.latency.processors import LATENCY_PROCESSORS


def compute_latency(mapping):
    return _compute_latency(mapping, 0)[0]


def _compute_latency(mapping, top_idx: int):
    next_top_idx = top_idx
    result = []
    for node in mapping:
        next_top_idx += 1

        if node['type'] not in LATENCY_PROCESSORS.keys():
            continue

        children_latencies = [
            _compute_latency(branch, next_top_idx)
            for branch in node['branches']
        ]

        result.append(
            LATENCY_PROCESSORS[node['type']](top_idx, children_latencies)
        )
    return result

