from collections.abc import Mapping

import islpy as isl

from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
from pytimeloop.looptree.mapping_utilities import *


def get_total_accesses(accesses: Mapping):
    result = {}
    for k, v in accesses.items():
        sum = get_sum_of_pw_qpolynomial(v)
        if sum.is_nan():
            result[k] = 0
        else:
            result[k] = sum.to_python()

    return result


def reads_and_writes_from_fill_by_parent(fills: Mapping, mapping, workload):
    mapping = mapping['nodes']
    dspace_id_to_name = workload.data_space_id_to_name()
    einsum_id_to_name = workload.einsum_id_to_name()

    reads = {}
    writes = {}

    parent_buffers = get_parent_buffers(mapping, workload)

    einsums_with_complete_mappings = get_einsums_with_complete_mappings(mapping)

    for (buffer_id, dspace_id, einsum_id), (tags, fill) in fills.items():
        dspace_name = dspace_id_to_name[dspace_id]
        einsum_name = einsum_id_to_name[einsum_id]
        if einsum_name not in einsums_with_complete_mappings:
            continue
        parent_buffer = parent_buffers[(buffer_id, dspace_name, einsum_name)]
        if parent_buffer is not None:
            if dspace_id in workload.tensors_written_by_einsum(einsum_id):
                writes[(parent_buffer, dspace_name, einsum_name)] = fill
                # TODO: first read elision
                reads[(parent_buffer, dspace_name, einsum_name)] = fill
            elif dspace_id in workload.tensors_read_by_einsum(einsum_id):
                reads[(parent_buffer, dspace_name, einsum_name)] = fill

    return reads, writes


def reads_and_writes_from_fill_by_peer(fills: Mapping, mapping, workload):
    mapping = mapping['nodes']
    dspace_id_to_name = workload.data_space_id_to_name()
    einsum_id_to_name = workload.einsum_id_to_name()

    reads = {}
    writes = {}

    einsums_with_complete_mappings = get_einsums_with_complete_mappings(mapping)

    for (buffer_id, dspace_id, einsum_id), (tags, fill) in fills.items():
        einsum_name = einsum_id_to_name[einsum_id]
        dspace_name = dspace_id_to_name[dspace_id]
        if einsum_name not in einsums_with_complete_mappings:
            continue

        reads[(buffer_id, dspace_name, einsum_name)] = fill
        writes[(buffer_id, dspace_name, einsum_name)] = fill

    return reads, writes


def get_parent_buffers(mapping, workload):
    parent_buffers = {}
    for path in get_paths(mapping):
        leaf = path[-1]
        einsum_name = leaf['einsum']
        einsum_id = workload.einsum_name_to_id()[einsum_name]

        dspace_to_top_buffer = {}
        for node in path:
            if node['type'] == 'storage':
                for dspace in node['dspace']:
                    key = (node['target'], dspace, einsum_name)
                    if dspace in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace]
                    else:
                        parent_buffers[key] = None
                    dspace_to_top_buffer[dspace] = node['target']
            elif node['type'] == 'compute':
                for dspace in workload.tensors_read_by_einsum(einsum_id):
                    dspace = workload.data_space_id_to_name()[dspace]
                    key = (node['target'], dspace, einsum_name)
                    if dspace in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace]
                for dspace in workload.tensors_written_by_einsum(einsum_id):
                    dspace = workload.data_space_id_to_name()[dspace]
                    key = (node['target'], dspace, einsum_name)
                    if dspace in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace]

    return parent_buffers