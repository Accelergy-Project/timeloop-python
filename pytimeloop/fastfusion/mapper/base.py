from copy import deepcopy
from functools import partial, reduce
from itertools import chain, combinations, permutations, product
from operator import mul

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from bindings.looptree import LooptreeWorkloadDependencyAnalyzer

from .shape_subspace import ShapeSubspace
from .stepped_model import Stats, SteppedModel, SteppedModelState


def mapper(config, spec, workload, name_of_einsum_to_eval, tmp_path):
    einsum_name_to_id = workload.einsum_name_to_id()
    id_of_einsum_to_eval = einsum_name_to_id[name_of_einsum_to_eval]

    bindings, max_spatial = get_hardware_levels(spec.architecture)

    ranks = workload.einsum_ospace_dimensions(id_of_einsum_to_eval)
    tensors = (
        workload.tensors_read_by_einsum(id_of_einsum_to_eval)
        |
        workload.tensors_written_by_einsum(id_of_einsum_to_eval)
    )

    # Shape is given as *inclusive* (min, max) by workload
    einsum_shape = {
        rank_id: workload.get_rank_shape(rank_id)[1]+1 for rank_id in ranks
    }

    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    model = SteppedModel(config, spec, bindings, workload, analyzer)
    model.call_accelergy(tmp_path)
    state = SteppedModelState()
    model.initialize(state, 0, id_of_einsum_to_eval, list(tensors))

    def step_back_model():
        model.step_back()

    def final_model(level, state, temporal_loops, spatial_loops, retained_tensors):
        model.add_compute(state,
                          level,
                          name_of_einsum_to_eval,
                          temporal_loops,
                          spatial_loops)
        return model.run(state)

    def partial_model(level, state, temporal_loops, spatial_loops, retained_tensors):
        model.add_storage(state,
                          level,
                          temporal_loops,
                          spatial_loops,
                          retained_tensors)
        return Stats()



    cur_mapper = None
    for hw_level in reversed(range(1, len(bindings))):
        if cur_mapper is None:
            cur_mapper = ExhaustiveLevelMapper(hw_level,
                                               ranks,
                                               tensors,
                                               max_spatial=max_spatial[hw_level],
                                               can_bypass=False,
                                               lower_mapper=None,
                                               partial_model=partial(final_model,
                                                                     level=hw_level),
                                               step_back_model=step_back_model)
        else:
            cur_mapper = ExhaustiveLevelMapper(hw_level,
                                               ranks,
                                               tensors,
                                               max_spatial=max_spatial[hw_level],
                                               can_bypass=True,
                                               lower_mapper=cur_mapper,
                                               partial_model=partial(partial_model,
                                                                     level=hw_level),
                                               step_back_model=step_back_model)

    for result in cur_mapper.run(einsum_shape, state):
        pass


def get_hardware_levels(arch):
    bindings = {}
    fanout = {}
    for node in arch['nodes']:
        bindings_id = len(bindings)
        bindings[bindings_id] = node['name']
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
    return bindings, fanout


class ExhaustiveLevelMapper:
    def __init__(self,
                 hw_level: str,
                 ranks,
                 tensors,
                 can_bypass,
                 lower_mapper,
                 partial_model,
                 step_back_model,
                 max_spatial=(1,),
                 max_capacity=None,
                 mapping_filter=None,
                 stats_filter=None):
        self.hw_level = hw_level
        self.ranks = ranks
        self.tensors = tensors
        self.can_bypass = can_bypass
        self.lower_mapper = lower_mapper
        self.mapping_filter = mapping_filter
        self.stats_filter = stats_filter
        self.partial_model = partial_model
        self.step_back_model = step_back_model
        self.max_spatial = max_spatial
        self.max_capacity = max_capacity

    def run(self, rank_shapes, state):
        if self.can_bypass:
            bypass_choices = chain.from_iterable(
                combinations(self.tensors, r) for r in range(1, len(self.tensors))
            )
        else:
            bypass_choices = [self.tensors]

        n_spatial = len(self.max_spatial)
        for bypassing in bypass_choices:
            temporal_ranks_choices = permutations(self.ranks, len(self.ranks))
            spatial_ranks_choices = permutations(self.ranks)
            if reduce(mul, self.max_spatial, 1) == 1:
                spatial_ranks_choices = [[]]
            else:
                spatial_ranks_choices = product(spatial_ranks_choices, repeat=n_spatial)
            for temporal_ranks in temporal_ranks_choices:
                for spatial_ranks in spatial_ranks_choices:
                    n_temporal_ranks = len(temporal_ranks)
                    n_spatial_ranks = tuple(len(ranks) for ranks in spatial_ranks)
                    all_ranks = (
                        list(temporal_ranks) 
                        +
                        sum((list(ranks) for ranks in spatial_ranks), start=[])
                    )
                    tile_shape_subspace = ShapeSubspace(rank_shapes, all_ranks)
                    tile_shape_iterator = iter(tile_shape_subspace)
                    for tile_shape, leftover_rank_shapes in tile_shape_iterator:
                        temporal_tile_shape = tile_shape[:n_temporal_ranks]
                        start = n_temporal_ranks
                        spatial_tile_shapes = []
                        for num_ranks in n_spatial_ranks:
                            spatial_tile_shapes.append(tile_shape[start:start+num_ranks])
                            start += num_ranks

                        temporal_loops = list(zip(temporal_ranks, temporal_tile_shape))
                        if not self.check_mapping(temporal_loops, tile_shape, bypassing):
                            continue

                        spatial_loops = [
                            list(zip(ranks, spatial_tile_shape))
                            for ranks, spatial_tile_shape
                            in zip(spatial_ranks, spatial_tile_shapes)
                        ]

                        new_state = deepcopy(state)
                        stats = self.partial_model(state=new_state,
                                                   temporal_loops=temporal_loops,
                                                   spatial_loops=spatial_loops,
                                                   retained_tensors=bypassing)

                        if self.lower_mapper is not None:
                            for stats in self.lower_mapper.run(leftover_rank_shapes, new_state):
                                invalid_spatial = any(
                                    spatial_fanout > max_fanout
                                    for spatial_fanout, max_fanout
                                    in zip(stats.spatial[self.hw_level], self.max_spatial)
                                )
                                if invalid_spatial:
                                    break

                                invalid_capacity = (
                                    self.max_capacity is not None
                                    and
                                    stats.capacity[self.hw_level] > self.max_capacity
                                )
                                if invalid_capacity:
                                    tile_shape_iterator.skip_current_rank_iteration()
                                    break

                                yield stats
                        else:
                            invalid_spatial = any(
                                spatial_fanout > max_fanout
                                for spatial_fanout, max_fanout
                                in zip(stats.spatial[self.hw_level], self.max_spatial)
                            )
                            if invalid_spatial:
                                continue

                            invalid_capacity = (
                                self.max_capacity is not None
                                and
                                stats.capacity[self.hw_level] > self.max_capacity
                            )
                            if invalid_capacity:
                                tile_shape_iterator.skip_current_rank_iteration()

                            yield stats

    def check_mapping(self, temporal_loops, tile_shape, bypassing):
        if self.mapping_filter is None:
            return True
        return self.mapping_filter(temporal_loops, tile_shape, bypassing)

    def check_stats(self, total_stats):
        if self.stats_filter is None:
            return True
        return self.stats_filter(total_stats)