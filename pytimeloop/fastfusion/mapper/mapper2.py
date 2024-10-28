from collections import defaultdict
from dataclasses import dataclass
from itertools import product, permutations
from functools import reduce
from operator import or_, mul
from pathlib import Path

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from bindings.looptree import (
    LooptreeWorkload,
    LooptreeWorkloadDependencyAnalyzer
)

from pytimeloop.looptree.energy import gather_actions, compute_energy_from_actions
from pytimeloop.fastfusion.fastmodel import compile_mapping, LooptreeOutput
from pytimeloop.fastfusion.mapper.shape_subspace import ShapeSubspace
from pytimeloop.fastfusion.pareto import nameloop2col
from pytimeloop.fastfusion.compatibility import Compatibility, Loop, TensorTiling

from pytimeloop.timeloopfe.v4 import Ert
from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose

class LinearMapping:
    def __init__(self):
        self.mapping = []

    def __iter__(self):
        return iter(self.mapping)

    def __getitem__(self, key):
        return self.mapping[key]

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return repr(self.mapping)

    def copy(self):
        lm = LinearMapping()
        lm.mapping = self.mapping.copy()
        return lm

    def add_compute(self, einsum_name, target):
        self.mapping.append({
            'type': 'compute',
            'einsum': einsum_name,
            'target': target
        })

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

    def add_sequential(self, idx=None):
        node = {'type': 'sequential'}
        if idx is None:
            self.mapping.append(node)
        else:
            self.mapping.insert(idx, node)

    def add_pipeline(self):
        self.mapping.append({'type': 'pipeline'})

    def add_storage(self, target, dspaces, idx=None):
        node = {
            'type': 'storage',
            'target': target,
            'dspace': dspaces
        }
        if idx is None:
            self.mapping.append(node)
        else:
            self.mapping.insert(idx, node)


@dataclass
class MacArrayConstraint:
    array_shape_in_parallel_dimension: str
    array_shape_in_reduced_dimension: str

    weight_tensor: dict[str, str]
    parallel_rank: dict[str, str]
    reduced_rank: dict[str, str]


def mapper(config,
           mac_array_constraint: MacArrayConstraint,
           spec,
           tmp_path,
           verbose_stream=None):

    workload = LooptreeWorkload.parse_cfg(config.root['problem'])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    einsum_id_to_name = workload.einsum_id_to_name()
    rank_name_to_id = workload.dimension_name_to_id()
    tensor_name_to_id = workload.data_space_name_to_id()

    mac_parallel_shape = mac_array_constraint.array_shape_in_parallel_dimension
    mac_reduced_shape = mac_array_constraint.array_shape_in_reduced_dimension

    einsum_name_to_parallel_rank_name = mac_array_constraint.parallel_rank
    einsum_name_to_reduced_rank_name = mac_array_constraint.reduced_rank

    bindings, max_fanout, max_capacity = get_hardware_levels(spec.architecture)

    einsum_name_to_id = workload.einsum_name_to_id()

    if isinstance(tmp_path, Path):
        tmp_path = str(tmp_path)
    call_accelergy_verbose(spec, tmp_path)
    ert_dict = yaml.load(Path(tmp_path) / 'ERT.yaml')
    ert = Ert(ert_dict['ERT'])

    data = {}
    for einsum_id in einsum_name_to_id.values():
        data[einsum_id] = defaultdict(lambda: defaultdict(lambda: list()))
        tensors = (
            workload.tensors_read_by_einsum(einsum_id)
            |
            workload.tensors_written_by_einsum(einsum_id)
        )
        intermediate_tensors = tensors & get_intermediate_tensors(workload)

        einsum_name = einsum_id_to_name[einsum_id]
        mac_parallel_rank_name = einsum_name_to_parallel_rank_name[einsum_name]
        mac_parallel_rank_id = rank_name_to_id[mac_parallel_rank_name]
        mac_reduced_rank_name = einsum_name_to_reduced_rank_name[einsum_name]
        mac_reduced_rank_id = rank_name_to_id[mac_reduced_rank_name]

        weight_tensor_name = mac_array_constraint.weight_tensor[einsum_name]
        weight_tensor_id = tensor_name_to_id[weight_tensor_name]
        weight_ranks = analyzer.einsum_dims_relevant_to_tensor(einsum_id,
                                                               weight_tensor_id)
        other_weight_ranks = \
            weight_ranks - {mac_parallel_rank_id, mac_reduced_rank_id}
        all_ranks = workload.einsum_ospace_dimensions(einsum_id)
        non_weight_ranks = set(all_ranks) - weight_ranks

        tensor_to_relevant_ranks = {
            tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
            for tensor in tensors
        }

        einsum_shape = {
            rank_id: workload.get_rank_shape(rank_id)[1]+1 for rank_id in all_ranks
        }


        count = 0
        mapping = LinearMapping()
        mapping.add_storage(0, tensors-intermediate_tensors)
        top_level_ranks = reduce(
            or_,
            (tensor_to_relevant_ranks[t] for t in intermediate_tensors),
            set()
        )
        for partial_mapping in make_top_loops(mapping, top_level_ranks):
            for partial_mapping in place_fusion_level(partial_mapping,
                                                      intermediate_tensors,
                                                      tensor_to_relevant_ranks):
                for partial_mapping in make_pe_spatial_fors(partial_mapping,
                                                            all_ranks):
                    for partial_mapping in make_pe_temporal_fors(partial_mapping,
                                                                 all_ranks):
                        for partial_mapping in place_pe_level(partial_mapping,
                                                              tensors,
                                                              tensor_to_relevant_ranks):
                            for partial_mapping in make_mac_level_loops(partial_mapping,
                                                                        einsum_id,
                                                                        mac_parallel_rank_id,
                                                                        mac_parallel_shape,
                                                                        mac_reduced_rank_id,
                                                                        mac_reduced_shape,
                                                                        non_weight_ranks,
                                                                        other_weight_ranks):
                                _, compiled_results = compile_mapping(partial_mapping,
                                                                      workload,
                                                                      analyzer)
                                for shape, res in explore_tile_shape(partial_mapping,
                                                                     einsum_shape,
                                                                     compiled_results,
                                                                     max_capacity,
                                                                     max_fanout):
                                    process_result(res,
                                                   shape,
                                                   data[einsum_id],
                                                   einsum_id,
                                                   partial_mapping,
                                                   bindings,
                                                   workload,
                                                   ert)


def make_top_loops(mapping: LinearMapping, ranks):
    original = mapping
    for r in range(len(ranks)+1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_temporal(r)
            yield mapping


def place_fusion_level(mapping: LinearMapping,
                       intermediate_tensors,
                       tensor_to_relevant_ranks):
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
        untiled = True
        for i, node in enumerate(mapping[top_idx:], start=top_idx):
            if node['type'] == 'temporal':
                untiled = False
                rank_id = node['rank']
                is_relevant = rank_id in relevant_ranks
                if last_is_relevant and not is_relevant:
                    # Choice 1: fused
                    tensor_choices.append((i, 1))
                last_is_relevant = is_relevant
        if last_is_relevant:
            tensor_choices.append((len(mapping), 1))

        # If untiled, another choice: unfused
        if untiled:
            tensor_choices.append((len(mapping), 0))

        all_tensor_choices.append(tensor_choices)

    original = mapping.copy()
    for choices in product(*all_tensor_choices):
        if not any(c == len(mapping) for (c, level) in choices):
            continue
        mapping = original.copy()
        for choice, tensor in sorted(zip(choices, intermediate_tensors),
                                     key=lambda pair: pair[0],
                                     reverse=True):
            idx, level = choice
            mapping.add_sequential(idx)
            mapping.add_storage(level, {tensor}, idx=idx)
        yield mapping


def make_pe_spatial_fors(mapping, ranks):
    original = mapping.copy()
    for r in range(len(ranks)+1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_spatial(r)
            yield mapping


def make_pe_temporal_fors(mapping, ranks):
    original = mapping.copy()
    for r in range(len(ranks)+1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_spatial(r)
            yield mapping


def place_pe_level(mapping, tensors, tensor_to_relevant_ranks):
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
                    tensor_choices.append((i, 2))
                last_is_relevant = is_relevant
        if last_is_relevant:
            tensor_choices.append((len(mapping), 2))
        all_tensor_choices.append(tensor_choices)

    original = mapping.copy()
    for choices in product(*all_tensor_choices):
        mapping = original.copy()
        for choice, tensor in sorted(zip(choices, tensors),
                                     key=lambda pair: pair[0],
                                     reverse=True):
            idx, level = choice
            mapping.add_storage(level, {tensor}, idx=idx)
        yield mapping


def make_mac_level_loops(mapping,
                         einsum_id,
                         parallel_rank,
                         parallel_rank_shape,
                         reduced_rank,
                         reduced_rank_shape,
                         non_weight_ranks,
                         other_weight_ranks):
    mapping = mapping.copy()
    for rank in other_weight_ranks:
        mapping.add_temporal(rank, 1)
    mapping.add_temporal(parallel_rank, parallel_rank_shape)
    mapping.add_temporal(reduced_rank, reduced_rank_shape)
    for rank in non_weight_ranks:
        mapping.add_temporal(rank, 1)
    mapping.add_spatial(parallel_rank, 1)
    mapping.add_spatial(reduced_rank, 1)
    mapping.add_compute(einsum_id, 3)
    yield mapping


def explore_tile_shape(mapping,
                       rank_shapes,
                       compiled_result,
                       max_capacity,
                       max_fanout,
                       only_count=False):
    ranks = []
    for node in mapping:
        if node['type'] in ['temporal', 'spatial'] and 'tile_shape' not in node:
            ranks.append(node['rank'])

    num_tile_shapes = 0

    shape_subspace = iter(ShapeSubspace(rank_shapes, ranks))
    for shape in shape_subspace:
        num_tile_shapes += 1
        if only_count:
            continue

        result = LooptreeOutput()
        result.ops = call_with_arg(compiled_result.ops, shape)
        result.temporal_steps = call_with_arg(compiled_result.temporal_steps, shape)
        result.fanout = call_with_arg(compiled_result.fanout, shape)
        result.occupancy = call_with_arg(compiled_result.occupancy, shape)
        result.fills_by_parent = call_with_arg(compiled_result.fills_by_parent, shape)

        skip = False

        total_capacity = defaultdict(lambda: 0)
        for (level, _), capacity in result.occupancy.items():
            total_capacity[level] += capacity
        for level, capacity in total_capacity.items():
            if level in max_capacity and capacity > max_capacity[level]:
                skip = True
                break

        if skip == True:
            shape_subspace.skip_current_rank_iteration()
            continue

        for level, fanout in result.fanout.items():
            if level in max_fanout:
                invalid_spatial = any(
                    spatial_fanout_in_dim > max_fanout_in_dim
                    for spatial_fanout_in_dim, max_fanout_in_dim
                    in zip(fanout, max_fanout[level])
                )
                # if invalid_spatial:
                #     skip = True
                #     break

        if skip == True:
            shape_subspace.skip_current_rank_iteration()
            continue

        yield shape, result
    return num_tile_shapes


def process_result(result,
                   shape,
                   compatibility_to_df,
                   einsum_id,
                   mapping,
                   bindings,
                   workload,
                   ert):
    actions = gather_actions(result,
                             {'type': 'fused', 'nodes': mapping},
                             workload,
                             bindings,
                             is_path=True)
    energy = compute_energy_from_actions(actions, ert)
    energy = sum(energy.values())

    target_nloops_to_tensors = defaultdict(lambda: list())
    compat_tiling = defaultdict(lambda: list())
    cur_idx = 0
    cur_loops = []
    for node in mapping:
        if node['type'] == 'temporal':
            if 'tile_shape' in node:
                tile_shape = node['tile_shape']
            else:
                tile_shape = shape[cur_idx]
                cur_idx += 1
            cur_loops.append(Loop(node['rank'], tile_shape, False))
        elif node['type'] == 'spatial':
            if 'tile_shape' in node:
                tile_shape = node['tile_shape']
            else:
                tile_shape = shape[cur_idx]
                cur_idx += 1
            cur_loops.append(Loop(node['rank'], tile_shape, False))
        elif node['type'] == 'storage':
            dspaces = node['dspace']
            target = node['target']
            tensor_tiling = TensorTiling(target,
                                         tuple(cur_loops))
            for tensor in dspaces:
                if tensor not in compat_tiling:
                    compat_tiling[tensor] = tensor_tiling

                target_nloops_to_tensors[(target, len(cur_loops))].append(tensor)

    compatibility = Compatibility(
        tiling=compat_tiling,
        einsum_id=einsum_id
    )

    df = compatibility_to_df[compatibility]
    df['Latency'].append(result.temporal_steps[einsum_id])
    df['Energy'].append(energy)
    # Store PE spatial utilization
    df['PE_Utilization'].append(result.fanout[3])  # bindings[3] == 'PE'
    # Store footprints
    for (target, nloops), tensors in target_nloops_to_tensors.items():
        key = nameloop2col(target, nloops)
        lis = df[key]
        lis += [0]*(len(df['Latency'])-len(lis))  # lis may be added just now. Need to be padded
        for tensor in tensors:
            lis[-1] += result.occupancy[(target, tensor)]


def get_intermediate_tensors(workload: LooptreeWorkload):
    result = set()
    for einsum in workload.einsum_id_to_name():
        written_tensors = workload.tensors_written_by_einsum(einsum)
        for tensor in written_tensors:
            reader_einsums = workload.reader_einsums(tensor)
            for reader in reader_einsums:
                if reader in workload.einsum_id_to_name():
                    result.add(tensor)
                    break

    return result


def get_hardware_levels(arch):
    bindings = {}
    fanout = {}
    max_capacity = {}
    for node in arch['nodes']:
        bindings_id = len(bindings)
        bindings[bindings_id] = node['name']
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
        attribute = node.attributes
        if 'width' in attribute and 'depth' in attribute:
            width = attribute.width
            depth = attribute.depth
            datawidth = attribute.datawidth
            if all(x is not None for x in (width, depth, datawidth)):
                max_capacity[bindings_id] = \
                    attribute.width * attribute.depth / attribute.datawidth
    return bindings, fanout, max_capacity


def call_with_arg(f, arg):
    if isinstance(next(iter(f.values())), tuple):
        return { k: (v[0], v[1](*arg)) for k, v in f.items() }
    else:
        return { k: v(*arg) for k, v in f.items() }


def count(it):
    count = 0
    for _ in it:
        count += 1
    return count
