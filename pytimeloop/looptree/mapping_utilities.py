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


def get_leaves(mapping):
    for node in mapping:
        if node['type'] in ['pipeline', 'sequential']:
            for child in node['branches']:
                yield from get_leaves(child)
        elif node['type'] == 'compute':
            yield node


def get_einsums_with_complete_mappings(mapping):
    einsums_with_complete_mappings = set()
    for compute_node in get_leaves(mapping):
        if 'incomplete' not in compute_node:
            einsums_with_complete_mappings.add(compute_node['einsum'])
        if 'incomplete' in compute_node and not compute_node['incomplete']:
            einsums_with_complete_mappings.add(compute_node['einsum'])
    return einsums_with_complete_mappings
