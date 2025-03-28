from collections import defaultdict

from pytimeloop.isl.singular import get_value_from_singular_qpolynomial
from pytimeloop.looptree.accesses import (
    reads_and_writes_from_fill_by_parent,
    reads_and_writes_from_fill_by_peer
)
from pytimeloop.looptree.latency.processors import LATENCY_PROCESSORS
from pytimeloop.looptree.des import LooptreeOutput

from bindings.looptree import SpatialTag


def get_latency(looptree_results: LooptreeOutput,
                mapping,
                workload,
                arch,
                bindings):
    comp_latency = compute_latency(mapping,
                                   looptree_results.temporal_steps,
                                   workload)
    mem_latency = memory_latency(looptree_results,
                                 arch,
                                 mapping,
                                 workload,
                                 bindings)
    overall_latency = max(comp_latency, max(mem_latency.values()))
    return overall_latency, comp_latency, mem_latency


def compute_latency(mapping, temporal_steps, workload):
    return get_value_from_singular_qpolynomial(
        _compute_latency(mapping.nodes, 0, temporal_steps, workload)[1]
    ).to_python()


def memory_latency(looptree_results: LooptreeOutput,
                   arch,
                   mapping,
                   workload,
                   bindings):
    reads, writes = reads_and_writes_from_fill_by_parent(
        looptree_results.fills,
        looptree_results.reads_to_parent,
        mapping,
        workload,
        per_unit=True
    )

    peer_reads, peer_writes = reads_and_writes_from_fill_by_peer(
        looptree_results.reads_to_peer,
        mapping,
        workload,
        per_unit=True
    )

    component_to_read_writes = defaultdict(lambda: [None, None])
    for level, component in bindings.items():
        read_count = sum(reads[key] for key in reads if key[0] == level)
        read_count += sum(peer_reads[key]
                          for key in peer_reads if key[0] == level)
        write_count = sum(writes[key] for key in writes if key[0] == level)
        write_count += sum(peer_writes[key]
                           for key in peer_writes if key[0] == level)
        if component not in component_to_read_writes:
            component_to_read_writes[component][0] = read_count
            component_to_read_writes[component][1] = write_count
        else:
            component_to_read_writes[component][0] += read_count
            component_to_read_writes[component][1] += write_count

    component_latency = {}
    bandwidths = get_bandwidth(arch)
    for component, (reads, writes) in component_to_read_writes.items():
        read_bw, write_bw, shared_bw = bandwidths[component]

        # For numerical stability
        read_bw += 1e-8
        write_bw += 1e-8
        shared_bw += 1e-8

        # All shared bw for writing
        write_latency = writes / (write_bw + shared_bw)
        read_latency = reads / read_bw
        if write_latency >= read_latency:
            component_latency[component] = write_latency
            continue
        # All shared bw for reading
        write_latency = writes / write_bw
        read_latency = reads / (read_bw + shared_bw)
        if read_latency >= write_latency:
            component_latency[component] = read_latency
            continue
        # Shared bw shared for reading and writing
        component_latency[component] = (
            (reads + writes)
            / 
            (read_bw + write_bw + shared_bw)
        )
    return component_latency


def get_bandwidth(arch):
    component_bandwidths = {}
    for node in arch['nodes']:
        attributes = node.attributes
        n_rd_ports = attributes.get('n_rd_ports', 0)
        n_wr_ports = attributes.get('n_wr_ports', 0)
        n_rdwr_ports = attributes.get('n_rdwr_ports', 0)
        if n_rd_ports + n_wr_ports + n_rdwr_ports < 1:
            n_rdwr_ports = 1

        width = attributes['width']
        datawidth = attributes['datawidth']
        width_in_words = width/datawidth

        component_bandwidths[node['name']] = [
            n_rd_ports*width_in_words,
            n_wr_ports*width_in_words,
            n_rdwr_ports*width_in_words
        ]
    return component_bandwidths


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