from copy import deepcopy
from functools import reduce
from itertools import chain, combinations, permutations, product
from operator import mul

from pytimeloop.fastfusion.mapper.shape_subspace import ShapeSubspace

from .helper import gather_relevant_boundary_idxs


class ExhaustiveLevelMapper:
    def __init__(self,
                 hw_level: str,
                 ranks,
                 tensors,
                 can_bypass,
                 lower_mapper,
                 partial_model,
                 step_back_model,
                 analyzer,
                 max_spatial=(1,),
                 max_capacity=None):
        self.hw_level = hw_level
        self.ranks = ranks
        self.tensors = tensors
        self.can_bypass = can_bypass
        self.lower_mapper = lower_mapper
        self.partial_model = partial_model
        self.step_back_model = step_back_model
        self.analyzer = analyzer
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
                all_tensor_choices = []
                for tensor_id in bypassing:
                    relevant_ranks = \
                        self.analyzer.einsum_dims_relevant_to_tensor(
                            state.id_of_einsum_to_eval,
                            tensor_id
                        )
                    all_tensor_choices.append(
                        (tensor_id, i) for i in
                        gather_relevant_boundary_idxs(temporal_ranks,
                                                      relevant_ranks)
                    )
                for retain_choices in product(*all_tensor_choices):
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

                            temporal_loops = tuple(zip(temporal_ranks, temporal_tile_shape))

                            spatial_loops = [
                                tuple(zip(ranks, spatial_tile_shape))
                                for ranks, spatial_tile_shape
                                in zip(spatial_ranks, spatial_tile_shapes)
                            ]

                            new_state = deepcopy(state)
                            stats = self.partial_model(state=new_state,
                                                    temporal_loops=temporal_loops,
                                                    spatial_loops=spatial_loops,
                                                    retained_tensors=retain_choices)

                            if self.lower_mapper is not None:
                                for stats in self.lower_mapper.run(leftover_rank_shapes, new_state):
                                    invalid_spatial = any(
                                        spatial_fanout > max_fanout
                                        for spatial_fanout, max_fanout
                                        in zip(stats.spatial[self.hw_level], self.max_spatial)
                                    )
                                    if invalid_spatial:
                                        break

                                    total_capacity = 0
                                    for (level, _), capacity in stats.capacity.items():
                                        if level == self.hw_level:
                                            total_capacity += capacity
                                    invalid_capacity = (
                                        self.max_capacity is not None
                                        and
                                        total_capacity > self.max_capacity
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

                                total_capacity = 0
                                for (level, _), capacity in stats.capacity.items():
                                    if level == self.hw_level:
                                        total_capacity += capacity
                                invalid_capacity = (
                                    self.max_capacity is not None
                                    and
                                    total_capacity > self.max_capacity
                                )
                                if invalid_capacity:
                                    tile_shape_iterator.skip_current_rank_iteration()

                                yield stats
