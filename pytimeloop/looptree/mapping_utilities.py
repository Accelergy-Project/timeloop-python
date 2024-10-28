def get_paths(mapping):
    cur_path = []
    for node in mapping:
        cur_path.append(node)
        if node['type'] in ['pipeline', 'sequential']:
            for child in node['branches']:
                for subpath in get_paths(child):
                    yield cur_path + subpath
        elif node['type'] == 'compute':
            yield cur_path.copy()


def get_leaves(mapping, is_path):
    if is_path:
        yield mapping[-1]
        return
    for node in mapping:
        if node['type'] in ['pipeline', 'sequential']:
            for child in node['branches']:
                yield from get_leaves(child, is_path)
        elif node['type'] == 'compute':
            yield node


def get_einsums_with_complete_mappings(mapping, workload, is_path):
    einsums_with_complete_mappings = set()
    for compute_node in get_leaves(mapping, is_path):
        einsum_name = compute_node['einsum']
        if isinstance(einsum_name, int):
            einsum_id = einsum_name
        else:
            einsum_id = workload.einsum_name_to_id()[einsum_name]
        if 'incomplete' not in compute_node:
            einsums_with_complete_mappings.add(einsum_id)
        if 'incomplete' in compute_node and not compute_node['incomplete']:
            einsums_with_complete_mappings.add(einsum_id)
    return einsums_with_complete_mappings
