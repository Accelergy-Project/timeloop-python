from collections import defaultdict
from functools import reduce
from operator import mul

from bindings.looptree import LooptreeWorkload

from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_paths
from pytimeloop.looptree.des import LooptreeOutput


def run_fastmodel(mapping,
                  id_of_einsum_to_eval,
                  workload: LooptreeWorkload,
                  analyzer):
    mapping = mapping['nodes']

    einsum_name_to_id = workload.einsum_name_to_id()
    einsum_id_to_name = workload.einsum_id_to_name()
    rank_name_to_id = workload.dimension_name_to_id()
    tensor_name_to_id = workload.data_space_name_to_id()

    name_of_einsum_to_eval = einsum_id_to_name[id_of_einsum_to_eval]
    for path in get_paths(mapping):
        if path[-1]['einsum'] == name_of_einsum_to_eval:
            mapping = path
            break


    tensors = (
        workload.tensors_read_by_einsum(id_of_einsum_to_eval)
        |
        workload.tensors_written_by_einsum(id_of_einsum_to_eval)
    )

    rank_groups = EquivalentGroups.from_workload(workload, analyzer)
    einsum_shape = {
        group_id: workload.get_rank_shape(next(iter(equiv_ranks)))
        for group_id, equiv_ranks in rank_groups.group_id_to_ranks.items()
    }
    # Shape is given as *inclusive* (min, max) by workload
    einsum_shape = {k: v[1]+1 for k, v in einsum_shape.items()}

    tensor_size = {
        tensor_id: workload.get_tensor_volume(tensor_id)
        for tensor_id in tensor_name_to_id.values()
    }
    original_tensor_size = tensor_size.copy()

    output = LooptreeOutput()

    latency = 1
    potential_tensor_access_multiplier = defaultdict(lambda: 1)
    actual_tensor_access_multiplier = defaultdict(lambda: 1)
    tensor_to_relevant_ranks = {
        tensor_id: analyzer.einsum_dims_relevant_to_tensor(id_of_einsum_to_eval,
                                                           tensor_id)
        for tensor_id in tensors
    }
    tensor_to_relevant_ranks = {
        tensor_id: {
            rank_groups.rank_to_group_id[rank] for rank in relevant_ranks
        }
        for tensor_id, relevant_ranks in tensor_to_relevant_ranks.items()
    }
    fanout = {}
    cur_fanout = [1]
    for node in mapping:
        if node['type'] == 'temporal':
            rank_name = node['rank']
            rank_id = rank_name_to_id[rank_name]
            group_id = rank_groups.rank_to_group_id[rank_id]

            tile_shape = node['tile_shape']
            factor = einsum_shape[group_id] // tile_shape
            einsum_shape[group_id] = tile_shape

            latency *= factor

            for tensor_id in tensors:
                relevant_ranks = tensor_to_relevant_ranks[tensor_id]
                if group_id in relevant_ranks:
                    actual_tensor_access_multiplier[tensor_id] = \
                        potential_tensor_access_multiplier[tensor_id]
                    tensor_size[tensor_id] //= factor
                else:
                    potential_tensor_access_multiplier[tensor_id] *= factor
        elif node['type'] == 'spatial':
            rank_name = node['rank']
            rank_id = rank_name_to_id[rank_name]
            group_id = rank_groups.rank_to_group_id[rank_id]

            tile_shape = node['tile_shape']
            factor = einsum_shape[group_id] // tile_shape
            einsum_shape[group_id] = tile_shape

            if 'spatial' not in node:
                spatial = 0
            else:
                spatial = node['spatial']

            if spatial >= len(cur_fanout):
                cur_fanout += [1]*(spatial + 1 - len(cur_fanout))
            cur_fanout[spatial] *= factor
        elif node['type'] == 'storage':
            target = node['target']
            tensor_names = node['dspace']
            for tensor_name in tensor_names:
                tensor_id = tensor_name_to_id[tensor_name]
                if tensor_id not in tensors:
                    continue

                if cur_fanout is None:
                    cur_fanout = [1]

                output.occupancy[(target, tensor_id)] = tensor_size[tensor_id]

                output.fills_by_parent[(target, tensor_id, id_of_einsum_to_eval)] = (
                    None,
                    (
                        original_tensor_size[tensor_id]
                        *
                        actual_tensor_access_multiplier[tensor_id]
                        *
                        reduce(mul, cur_fanout, 1)
                    )
                )

                fanout[target] = cur_fanout
                cur_fanout = [1]
        elif node['type'] == 'compute':
            target = node['target']
            for tensor_id in tensors:
                output.occupancy[(target, tensor_id)] = tensor_size[tensor_id]

                output.fills_by_parent[(target, tensor_id, id_of_einsum_to_eval)] = (
                    None,
                    (
                        original_tensor_size[tensor_id]
                        *
                        potential_tensor_access_multiplier[tensor_id]
                    )
                )
            fanout[target] = cur_fanout

    output.ops[id_of_einsum_to_eval] = \
        (None, workload.get_operation_space_volume(id_of_einsum_to_eval))

    output.temporal_steps[id_of_einsum_to_eval] = latency
    output.fanout = fanout

    return output