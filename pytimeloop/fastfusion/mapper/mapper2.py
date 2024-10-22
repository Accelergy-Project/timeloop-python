from collections import defaultdict, deque
from functools import partial
from itertools import product, permutations

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from bindings.looptree import (
    LooptreeWorkload,
    LooptreeWorkloadDependencyAnalyzer
)

from pytimeloop.fastfusion.fastmodel import compile_mapping


class LinearMapping:
    def add_compute(self, einsum_name):
        self.mapping.append({'type': 'compute', 'einsum': einsum_name})

    def add_temporal(self, rank_name, tile_shape=None):
        node = {'type': 'temporal', 'rank': rank_name}
        if tile_shape is not None:
            node['tile_shape'] = tile_shape
        self.mapping.append(node)

    def add_spatial(self, rank_name, tile_shape=None):
        node = {'type': 'spatial', 'rank': rank_name}
        if tile_shape is not None:
            node['tile_shape'] = tile_shape
        self.mapping.append(node)

    def add_sequential(self):
        self.mapping.append({'type': 'sequential'})

    def add_pipeline(self):
        self.mapping.append({'type': 'pipeline'})

    def add_storage(self, target, dspaces):
        self.mapping.append({
            'type': 'storage',
            'target': target,
            'dspace': dspaces
        })


def mapper(config, spec, tmp_path, verbose_stream=None):
    workload = LooptreeWorkload.parse_cfg(config.root['problem'])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    einsum_name_to_id = workload.einsum_name_to_id()
    for einsum_id in einsum_name_to_id.values():
        tensors = (
            workload.tensors_read_by_einsum(einsum_id)
            |
            workload.tensors_written_by_einsum(einsum_id)
        )
        intermediate_tensors = tensors & get_intermediate_tensors(workload)

        mapping = LinearMapping()
        for partial_mapping in make_top_loops(mapping, einsum_id, workload):
            for partial_mapping in place_fusion_level(partial_mapping,
                                                    intermediate_tensors):
                for partial_mapping in make_pe_spatial_fors(partial_mapping,
                                                            einsum_id,
                                                            workload):
                    for partial_mapping in make_pe_temporal_fors(partial_mapping,
                                                                einsum_id,
                                                                workload):
                        for partial_mapping in place_pe_level(partial_mapping,
                                                            tensors):
                            for partial_mapping in make_mac_level_loops(partial_mapping,
                                                                        einsum_id,
                                                                        parallel_rank,
                                                                        parallel_rank_shape,
                                                                        reduced_rank,
                                                                        reduced_rank_shape,
                                                                        non_weight_ranks,
                                                                        other_weight_ranks):
                                compiled_results = compile_mapping(partial_mapping)
                                explore_tile_shape(partial_mapping, compiled_results)


# Determine all relevant ranks for top loops
# Place fusion (GLB, but maybe PE as well?) level
# Add spatial-fors (for PE)
# Add temporal loops
# Add spatial-fors (for MAC)
# Add temporal loops (for MAC)


def make_top_loops(mapping: LinearMapping, einsum_id, workload):
    ranks = workload.einsum_ospace_dimensions(einsum_id)
    for r in range(len(ranks)+1):
        for ordered_ranks in permutations(ranks, r=r):
            for r in ordered_ranks:
                mapping.add_temporal_loop(r)
            yield mapping


def place_fusion_level(mapping: LinearMapping, intermediate_tensors):
    top_idx = 0
    for node in mapping:
        if node['type'] != 'storage':
            break
        else:
            top_idx += 1

    all_tensor_choices = []
    for tensor_id in intermediate_tensors:
        relevant_ranks = tensor_to_relevant_ranks[tensor_id]
        tensor_choices = []
        last_is_relevant = True
        for i, node in enumerate(mapping[top_idx:], start=top_idx):
            if node['type'] == 'temporal':
                rank_id = node['rank']
                is_relevant = rank_id in relevant_ranks
                if last_is_relevant and not is_relevant:
                    # Choice 1: fused
                    tensor_choices.append((i, 'GLB'))
                    # If untiled, choice 2: unfused
                    if i == top_idx:
                        tensor_choices.append((i, 'DRAM'))
                last_is_relevant = is_relevant
        all_tensor_choices.append(tensor_choices)

    for choices in product(*all_tensor_choices):
        if not any(c == len(mapping) for (c, level) in choices):
            continue
        for choice, tensor in sorted(zip(choices, intermediate_tensors),
                                     key=lambda pair: pair[0],
                                     reverse=True):
            idx, level = choice
            mapping.insert_sequential(idx)
            mapping.insert_storage(idx, level, tensor)
        yield mapping


def make_pe_spatial_fors(mapping, einsum_id, workload):
    ranks = workload.einsum_ospace_dimensions(einsum_id)
    for r in range(len(ranks)+1):
        for ordered_ranks in permutations(ranks, r=r):
            for r in ordered_ranks:
                mapping.add_spatial_loop(r)
            yield mapping


def make_pe_temporal_fors(mapping, einsum_id, workload):
    ranks = workload.einsum_ospace_dimensions(einsum_id)
    for r in range(len(ranks)+1):
        for ordered_ranks in permutations(ranks, r=r):
            for r in ordered_ranks:
                mapping.add_spatial_loop(r)
            yield mapping


def place_pe_level(mapping, tensors):
    all_tensor_choices = []
    for tensor_id in tensors:
        relevant_ranks = tensor_to_relevant_ranks[tensor_id]
        tensor_choices = []
        last_is_relevant = True
        for i, node in enumerate(mapping):
            if node['type'] == 'temporal':
                rank_id = node['rank']
                is_relevant = rank_id in relevant_ranks
                if last_is_relevant and not is_relevant:
                    tensor_choices.append((i, 'PE'))
                last_is_relevant = is_relevant
        all_tensor_choices.append(tensor_choices)

    for choices in product(*all_tensor_choices):
        for choice, tensor in sorted(zip(choices, tensors),
                                     key=lambda pair: pair[0],
                                     reverse=True):
            idx, level = choice
            mapping.insert_storage(idx, level, tensor)
        yield mapping


def make_mac_level_loops(mapping,
                         einsum_id,
                         parallel_rank,
                         parallel_rank_shape,
                         reduced_rank,
                         reduced_rank_shape,
                         non_weight_ranks,
                         other_weight_ranks):
    for rank in other_weight_ranks:
        mapping.add_temporal_loop(rank, 1)
    mapping.add_temporal_loop(parallel_rank, parallel_rank_shape)
    mapping.add_temporal_loop(reduced_rank, reduced_rank_shape)
    for rank in non_weight_ranks:
        mapping.add_temporal_loop(rank, 1)
    mapping.add_spatial_loop(parallel_rank, 1)
    mapping.add_spatial_loop(reduced_rank, 1)
    mapping.add_compute(einsum_id)
    yield mapping


def explore_tile_shape(mapping, compiled_results):
    pass
