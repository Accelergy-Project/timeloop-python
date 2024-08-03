from collections.abc import Mapping

from pytimeloop.isl.top import Context, isl, libc
from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial


def get_total_accesses(accesses: Mapping[(int, str), int]):
    result = {
        k: get_sum_of_pw_qpolynomial(v)
        for k, v in accesses.items()
    }

    return result


def reads_and_writes_from_fill(fills: Mapping, mapping, workload):
    mapping = mapping['nodes']
    dspace_id_to_name = workload.data_space_id_to_name()

    reads = {}
    writes = {}

    parent_buffers = get_parent_buffers(mapping)

    for (buffer_id, dspace_id, einsum_id), fill in fills.items():
        dspace_name = dspace_id_to_name[dspace_id]
        parent_buffer = parent_buffers[(buffer_id, dspace_name)]
        if parent_buffer is not None:
            if dspace_id in workload.tensors_written_by_einsum(einsum_id):
                writes[(parent_buffer, dspace_name)] = fill
                # TODO: first read elision
                reads[(parent_buffer, dspace_name)] = fill
            elif dspace_id in workload.tensors_read_by_einsum(einsum_id):
                reads[(parent_buffer, dspace_name)] = fill

    return reads, writes


def get_parent_buffers(mapping):
    return _get_parent_buffers(mapping, {})


def _get_parent_buffers(mapping, dspace_to_top_buffer):
    parent_buffers = {}

    for node in mapping:
        if node['type'] == 'storage':
            for dspace in node['dspace']:
                key = (node['target'], dspace)
                if dspace in dspace_to_top_buffer:
                    parent_buffers[key] = dspace_to_top_buffer[dspace]
                else:
                    parent_buffers[key] = None

                dspace_to_top_buffer[dspace] = node['target']

        if node['type'] in ['parallel', 'pipeline', 'sequential']:
            for child in node['branches']:
                parent_buffers |= _get_parent_buffers(
                    child,
                    dspace_to_top_buffer.copy()
                )

    return parent_buffers
