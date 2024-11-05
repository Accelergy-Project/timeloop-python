from collections import defaultdict
from dataclasses import dataclass
from itertools import product, permutations
from functools import reduce
from operator import or_, mul
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML(typ="safe")

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.looptree.energy import gather_actions, compute_energy_from_actions
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups

from pytimeloop.fastfusion.fastmodel import compile_mapping, LooptreeOutput
from pytimeloop.fastfusion.mapper.shape_subspace import ShapeSubspace
from pytimeloop.fastfusion.pareto import nameloop2col
from pytimeloop.fastfusion.sim import TensorStorage, Tiling, Loop
from pytimeloop.fastfusion.pareto import MAPPING

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
        self.mapping.append(
            {"type": "compute", "einsum": einsum_name, "target": target}
        )

    def add_temporal(self, rank_name, tile_shape=None):
        node = {"type": "temporal", "rank": rank_name}
        if tile_shape is not None:
            node["tile_shape"] = tile_shape
        self.mapping.append(node)

    def add_spatial(
        self,
        rank_name,
        tile_shape=None,
        tile_shape_constraint=None,
        factor_constraint=None,
    ):
        node = {"type": "spatial", "rank": rank_name}
        if tile_shape is not None:
            node["tile_shape"] = tile_shape
        if tile_shape_constraint is not None:
            node["tile_shape_constraint"] = tile_shape_constraint
        if factor_constraint is not None:
            node["factor_constraint"] = factor_constraint
        self.mapping.append(node)

    def add_sequential(self, idx=None):
        node = {"type": "sequential"}
        if idx is None:
            self.mapping.append(node)
        else:
            self.mapping.insert(idx, node)

    def add_pipeline(self):
        self.mapping.append({"type": "pipeline"})

    def add_storage(self, target, dspaces, idx=None):
        node = {"type": "storage", "target": target, "dspace": dspaces}
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


@dataclass
class PeArrayConstraint:
    array_shape: int


def _mapper_one_einsum(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    spec,
    explore_glb_uneven,
    explore_pe_uneven,
    einsum_id,
    energy_dict,
    verbose_stream=None,
):
    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    einsum_id_to_name = workload.einsum_id_to_name()
    rank_name_to_id = workload.dimension_name_to_id()
    tensor_name_to_id = workload.data_space_name_to_id()

    mac_parallel_shape = mac_array_constraint.array_shape_in_parallel_dimension
    mac_reduced_shape = mac_array_constraint.array_shape_in_reduced_dimension

    einsum_name_to_parallel_rank_name = mac_array_constraint.parallel_rank
    einsum_name_to_reduced_rank_name = mac_array_constraint.reduced_rank

    bindings, max_fanout, max_capacity = get_hardware_levels(spec.architecture)

    data = {}
    data[einsum_id] = defaultdict(lambda: defaultdict(lambda: list()))
    tensors = workload.tensors_read_by_einsum(
        einsum_id
    ) | workload.tensors_written_by_einsum(einsum_id)
    intermediate_tensors = tensors & get_intermediate_tensors(workload)

    einsum_name = einsum_id_to_name[einsum_id]
    mac_parallel_rank_name = einsum_name_to_parallel_rank_name[einsum_name]
    mac_parallel_rank_id = rank_name_to_id[mac_parallel_rank_name]
    mac_reduced_rank_name = einsum_name_to_reduced_rank_name[einsum_name]
    mac_reduced_rank_id = rank_name_to_id[mac_reduced_rank_name]

    weight_tensor_name = mac_array_constraint.weight_tensor[einsum_name]
    weight_tensor_id = tensor_name_to_id[weight_tensor_name]
    weight_ranks = analyzer.einsum_dims_relevant_to_tensor(einsum_id, weight_tensor_id)
    other_weight_ranks = weight_ranks - {mac_parallel_rank_id, mac_reduced_rank_id}
    all_ranks = workload.einsum_ospace_dimensions(einsum_id)
    non_weight_ranks = set(all_ranks) - weight_ranks

    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    einsum_shape = {
        rank_id: workload.get_rank_shape(rank_id)[1] + 1 for rank_id in all_ranks
    }

    count = 0
    mapping = LinearMapping()
    mapping.add_storage(0, tensors - intermediate_tensors)
    top_level_ranks = reduce(
        or_, (tensor_to_relevant_ranks[t] for t in intermediate_tensors), set()
    )
    mappings = defaultdict(list)
    for partial_mapping in make_top_loops(mapping, top_level_ranks):
        for partial_mapping in place_fusion_level(
            partial_mapping,
            intermediate_tensors,
            tensor_to_relevant_ranks,
            explore_glb_uneven,
        ):
            for partial_mapping in make_pe_spatial_fors(
                partial_mapping, all_ranks, pe_array_constraint
            ):
                for partial_mapping in make_pe_temporal_fors(
                    partial_mapping, all_ranks
                ):
                    for partial_mapping in place_pe_level(
                        partial_mapping,
                        tensors,
                        tensor_to_relevant_ranks,
                        explore_pe_uneven,
                    ):
                        for partial_mapping in make_mac_level_loops(
                            partial_mapping,
                            einsum_id,
                            mac_parallel_rank_id,
                            mac_parallel_shape,
                            mac_reduced_rank_id,
                            mac_reduced_shape,
                            non_weight_ranks,
                            other_weight_ranks,
                        ):
                            _, compiled_results = compile_mapping(
                                partial_mapping, workload, analyzer
                            )
                            for shape, res in explore_tile_shape(
                                partial_mapping,
                                einsum_shape,
                                compiled_results,
                                max_capacity,
                                max_fanout,
                            ):
                                count += 1
                                tiling, stats, fulltiling = process_result(
                                    res,
                                    shape,
                                    data[einsum_id],
                                    einsum_id,
                                    intermediate_tensors,
                                    partial_mapping,
                                    bindings,
                                    workload,
                                    energy_dict,
                                    equivalent_groups,
                                )
                                if count % 1e4 == 0:
                                    print(f"Count: {count}, fulltiling: {fulltiling}")
                                mappings[tiling].append(stats)
    return mappings


def mapper(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    explore_glb_uneven,
    explore_pe_uneven,
    spec,
    tmp_path,
    verbose_stream=None,
):

    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

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
    ert_dict = yaml.load(Path(tmp_path) / "ERT.yaml")
    ert = Ert(ert_dict["ERT"])
    energy_dict = ert.to_dict()

    data = {}
    per_einsum_args = [
        dict(
            einsum_id=einsum_id,
            config=config,
            pe_array_constraint=pe_array_constraint,
            mac_array_constraint=mac_array_constraint,
            explore_glb_uneven=explore_glb_uneven,
            explore_pe_uneven=explore_pe_uneven,
            spec=spec,
            energy_dict=energy_dict,
            verbose_stream=verbose_stream,
        )
        for einsum_id in einsum_name_to_id.values()
    ]

    from joblib import Parallel, delayed

    data = Parallel(n_jobs=1)(
        delayed(_mapper_one_einsum)(**args) for args in per_einsum_args
    )
    data = {
        einsum_id: mapping
        for einsum_id, mapping in zip(einsum_name_to_id.values(), data)
    }
    return data


def make_top_loops(mapping: LinearMapping, ranks):
    original = mapping
    for r in range(len(ranks) + 1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_temporal(r)
            yield mapping


def place_fusion_level(
    mapping: LinearMapping,
    intermediate_tensors,
    tensor_to_relevant_ranks,
    explore_uneven=True,
):
    top_idx = 0
    for node in mapping:
        if node["type"] != "storage":
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
            if node["type"] == "temporal":
                untiled = False
                if explore_uneven:
                    rank_id = node["rank"]
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
        for choice, tensor in sorted(
            zip(choices, intermediate_tensors), key=lambda pair: pair[0], reverse=True
        ):
            idx, level = choice
            mapping.add_sequential(idx)
            mapping.add_storage(level, {tensor}, idx=idx)
        yield mapping


def make_pe_spatial_fors(mapping, ranks, pe_array_constraint: PeArrayConstraint):
    original = mapping.copy()
    for r in range(len(ranks) + 1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_spatial(
                    r, factor_constraint=f"<={pe_array_constraint.array_shape}"
                )
            yield mapping


def make_pe_temporal_fors(mapping, ranks):
    original = mapping.copy()
    for r in range(len(ranks) + 1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_temporal(r)
            yield mapping


def place_pe_level(mapping, tensors, tensor_to_relevant_ranks, explore_uneven):
    all_tensor_choices = []
    for tensor_id in tensors:
        relevant_ranks = tensor_to_relevant_ranks[tensor_id]
        tensor_choices = []
        last_is_relevant = True
        if explore_uneven:
            for i, node in enumerate(mapping):
                if node["type"] == "temporal":
                    rank_id = node["rank"]
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
        for choice, tensor in sorted(
            zip(choices, tensors), key=lambda pair: pair[0], reverse=True
        ):
            idx, level = choice
            mapping.add_storage(level, {tensor}, idx=idx)
        yield mapping


def make_mac_level_loops(
    mapping,
    einsum_id,
    parallel_rank,
    parallel_rank_shape,
    reduced_rank,
    reduced_rank_shape,
    non_weight_ranks,
    other_weight_ranks,
):
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


def explore_tile_shape(
    mapping, rank_shapes, compiled_result, max_capacity, max_fanout, only_count=False
):
    ranks = []
    tile_constraints = []
    factor_constraints = []
    for node in mapping:
        if node["type"] in ["temporal", "spatial"] and "tile_shape" not in node:
            ranks.append(node["rank"])
            tile_constraint = []
            factor_constraint = []
            if "tile_constraint" in node:
                tile_constraint.append(node["tile_constraint"])
            if "factor_constraint" in node:
                factor_constraint.append(node["factor_constraint"])
            tile_constraints.append(tile_constraint)
            factor_constraints.append(factor_constraint)

    num_tile_shapes = 0

    shape_subspace = iter(
        ShapeSubspace(
            rank_shapes,
            ranks,
            tile_constraints=tile_constraints,
            factor_constraints=factor_constraints,
        )
    )
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

        invalid_spatial = False
        for level, fanout in result.fanout.items():
            if level in max_fanout:
                invalid_spatial = invalid_spatial or (
                    reduce(mul, fanout, 1) > reduce(mul, max_fanout[level], 1)
                )

        if skip == True:
            shape_subspace.skip_current_rank_iteration()
            continue

        if not invalid_spatial:
            yield shape, result
    return num_tile_shapes


import time


def process_result(
    result,
    shape,
    compatibility_to_df,
    einsum_id,
    intermediate_tensors,
    mapping,
    bindings,
    workload,
    energy_dict,
    equiv_groups: EquivalentGroups,
):
    t0 = time.time()
    actions = gather_actions(
        result, {"type": "fused", "nodes": mapping}, workload, bindings, is_path=True
    )
    t1 = time.time()
    energy = sum(  # - 40k ms
        energy_dict[comp_action] * counts for comp_action, counts in actions.items()
    )

    cur_idx = 0
    cur_loops = []
    tensors = []
    found_tensors = []
    reservations = {}
    found_intermediate_tensors = 0
    
    def record_backing_storage(dspace, target, n_loops):
        if dspace in found_tensors:
            return
        
        nonlocal found_intermediate_tensors
        tensors.append(TensorStorage(dspace, target, n_loops, 0))
        found_tensors.append(dspace)
        if dspace in intermediate_tensors:
            found_intermediate_tensors += 1

    def record_reservation(dspace, target, n_loops):
        reservations.setdefault((target, n_loops), 0)
        reservations[(target, n_loops)] += result.occupancy[(target, dspace)]

    for node in mapping:
        if node["type"] == "storage":
            for dspace in node["dspace"]:
                record_backing_storage(dspace, node["target"], len(cur_loops))
                record_reservation(dspace, node["target"], len(cur_loops))

        elif node["type"] == "spatial" or node["type"] == "temporal":
            if "tile_shape" in node:
                tile_shape = node["tile_shape"]
            else:
                tile_shape = shape[cur_idx]
                cur_idx += 1
            if found_intermediate_tensors < len(intermediate_tensors):
                cur_loops.append(
                    Loop(
                        str(equiv_groups.rank_to_group_id[node["rank"]]),
                        tile_shape,
                        node["type"] == "spatial",
                    )
                )

    fulltiling = []
    for node in mapping:
        if node["type"] == "storage":
            fulltiling.append(f"S({node['dspace']} in {node['target']})")
        elif node["type"] == "temporal":
            fulltiling.append(f"T{node['rank']} in {node.get('tile_shape', 1)}")
        elif node["type"] == "spatial":
            fulltiling.append(f"S{node['rank']} in {node.get('tile_shape', 1)}")

    t2 = time.time()

    tiling = Tiling(
        loops=tuple(cur_loops),
        tensors=frozenset(t for t in tensors if t.tensor_id in intermediate_tensors),
    )

    results = {}
    results["Latency"] = result.temporal_steps[einsum_id]
    results["Energy"] = energy
    # results["PE_Utilization"] = result.fanout[3][0]
    results[MAPPING] = {einsum_id: str(tiling)}
    for (storage_id, n_loops), size in reservations.items():
        key = nameloop2col(storage_id, n_loops)
        results.setdefault(key, 0)
        results[key] += size
    t3 = time.time()
    fulltiling.append(f"{result.fanout}")
    fulltiling.append(reservations)
    fulltiling.append(f"{results['Latency']}")
    fulltiling.append(f"{results['Energy']}")
    # print(f"{(t1-t0)*1e9:.2f} {(t2-t1)*1e9:.2f} {(t3-t2)*1e9:.2f}")
    return tiling, results, fulltiling


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
    for node in arch["nodes"]:
        bindings_id = len(bindings)
        bindings[bindings_id] = node["name"]
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
        attribute = node.attributes
        if "width" in attribute and "depth" in attribute:
            width = attribute.width
            depth = attribute.depth
            datawidth = attribute.datawidth
            if all(x is not None for x in (width, depth, datawidth)):
                max_capacity[bindings_id] = (
                    attribute.width * attribute.depth / attribute.datawidth
                )
    return bindings, fanout, max_capacity


def call_with_arg(f, arg):
    if isinstance(next(iter(f.values())), tuple):
        return {k: (v[0], v[1](*arg)) for k, v in f.items()}
    else:
        return {k: v(*arg) for k, v in f.items()}


def count(it):
    count = 0
    for _ in it:
        count += 1
    return count
