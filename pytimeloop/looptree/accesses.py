from collections import defaultdict
from collections.abc import Mapping
from numbers import Number

import islpy as isl

from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
from pytimeloop.looptree.mapping_utilities import *


def get_total_accesses(accesses: Mapping):
    result = {}
    for k, v in accesses.items():
        if isinstance(v, isl.PwQPolynomial):
            sum = get_sum_of_pw_qpolynomial(v)
            if sum.is_nan():
                result[k] = 0
            else:
                result[k] = sum.to_python()
        elif isinstance(v, Number):
            result[k] = v
        else:
            result[k] = v

    return result


def reads_and_writes_from_fill_by_parent(fills: Mapping,
                                         reads_to_parent,
                                         mapping,
                                         workload,
                                         is_path=False):
    mapping = mapping['nodes']
    dspace_id_to_name = workload.data_space_id_to_name()
    einsum_id_to_name = workload.einsum_id_to_name()

    reads = defaultdict(lambda: 0)
    writes = defaultdict(lambda: 0)

    parent_buffers = get_parent_buffers(mapping, workload, is_path)

    einsums_with_complete_mappings = get_einsums_with_complete_mappings(mapping, workload, is_path)

    compute_node = mapping[-1]
    assert compute_node["type"] == "compute"
    compute_target = compute_node["target"]

    for (buffer_id, dspace_id, einsum_id), (tags, fill) in fills.items():
        read_to_parent = reads_to_parent[(buffer_id, dspace_id, einsum_id)][1]
        dspace_name = dspace_id_to_name[dspace_id]
        einsum_name = einsum_id_to_name[einsum_id]
        if einsum_id not in einsums_with_complete_mappings:
            continue
        parent_buffer = parent_buffers[(buffer_id, dspace_id, einsum_id)]
        if parent_buffer is not None:
            if dspace_id in workload.tensors_written_by_einsum(einsum_id):
                writes[(parent_buffer, dspace_name, einsum_name)] += read_to_parent
                # TODO: first read elision
                reads[(parent_buffer, dspace_name, einsum_name)] += read_to_parent
            elif dspace_id in workload.tensors_read_by_einsum(einsum_id):
                reads[(parent_buffer, dspace_name, einsum_name)] += read_to_parent
        if buffer_id != compute_target:
            writes[(buffer_id, dspace_name, einsum_name)] += fill

    return reads, writes


def reads_and_writes_from_fill_by_peer(fills: Mapping, mapping, workload, is_path=False):
    mapping = mapping['nodes']
    dspace_id_to_name = workload.data_space_id_to_name()
    einsum_id_to_name = workload.einsum_id_to_name()

    reads = {}
    writes = {}

    einsums_with_complete_mappings = get_einsums_with_complete_mappings(mapping, workload, is_path)

    for (buffer_id, dspace_id, einsum_id), (tags, fill) in fills.items():
        einsum_name = einsum_id_to_name[einsum_id]
        dspace_name = dspace_id_to_name[dspace_id]
        if einsum_id not in einsums_with_complete_mappings:
            continue

        reads[(buffer_id, dspace_name, einsum_name)] = fill
        writes[(buffer_id, dspace_name, einsum_name)] = fill

    return reads, writes


def get_parent_buffers(mapping, workload, is_path):
    parent_buffers = {}
    if is_path:
        paths = [mapping]
    else:
        paths = get_paths(mapping)

    for path in paths:
        leaf = path[-1]
        einsum_name = leaf['einsum']
        if isinstance(einsum_name, int):
            einsum_id = einsum_name
        else:
            einsum_id = workload.einsum_name_to_id()[einsum_name]

        dspace_to_top_buffer = {}
        for node in path:
            if node['type'] == 'storage':
                for dspace in node['dspace']:
                    if isinstance(dspace, int):
                        dspace_id = dspace
                    else:
                        dspace_id = workload.data_space_name_to_id()[dspace]
                    key = (node['target'], dspace_id, einsum_id)
                    if dspace_id in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace_id]
                    else:
                        parent_buffers[key] = None
                    dspace_to_top_buffer[dspace_id] = node['target']
            elif node['type'] == 'compute':
                for dspace_id in workload.tensors_read_by_einsum(einsum_id):
                    key = (node['target'], dspace_id, einsum_id)
                    if dspace_id in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace_id]
                for dspace_id in workload.tensors_written_by_einsum(einsum_id):
                    key = (node['target'], dspace_id, einsum_id)
                    if dspace_id in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace_id]

    return parent_buffers