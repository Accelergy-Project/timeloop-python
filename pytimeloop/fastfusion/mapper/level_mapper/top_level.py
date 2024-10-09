from copy import deepcopy
from collections import defaultdict
from functools import reduce
from itertools import permutations, product
from operator import mul

from more_itertools import powerset

import pandas as pd

from pytimeloop.fastfusion.mapper.shape_subspace import ShapeSubspace
from pytimeloop.fastfusion.compatibility import OpCompatibility
from pytimeloop.fastfusion.mapper.stepped_model import SteppedModelState


class TopLevelMapper:
    def __init__(self,
                 hw_level: str,
                 ranks,
                 tensors,
                 fusable_tensors,
                 id_of_einsum_to_eval,
                 neighbors,
                 lower_mapper,
                 model,
                 partial_model,
                 step_back_model,
                 bits_per_word,
                 max_spatial=(1,),
                 max_capacity=None,
                 mapping_filter=None,
                 stats_filter=None):
        self.hw_level = hw_level
        self.ranks = frozenset(ranks)
        self.tensors = frozenset(tensors)
        self.fusable_tensors = fusable_tensors
        self.id_of_einsum_to_eval = id_of_einsum_to_eval
        self.neighbors = frozenset(neighbors)
        self.lower_mapper = lower_mapper
        self.mapping_filter = mapping_filter
        self.stats_filter = stats_filter
        self.model = model
        self.partial_model = partial_model
        self.step_back_model = step_back_model
        self.max_spatial = max_spatial
        self.max_capacity = max_capacity
        self.compatibility_to_df = defaultdict(lambda: defaultdict(lambda: list()))
        self.bits_per_word = bits_per_word

    def store_evaluation_result(self, fused_tensors, fused_loops, metrics):
        op_compatibility = OpCompatibility(
            fused_tensors=fused_tensors,
            fused_loops=fused_loops,
            ranks=self.ranks,
            tensors=self.tensors,
            neighbors=self.neighbors,
            einsum_id=self.id_of_einsum_to_eval
        )
        df = self.compatibility_to_df[op_compatibility]

        df['__Occupancy'].append(sum(metrics.capacity.values()))
        for tensor in self.tensors:
            df[f'__{tensor} Data Size'].append(self.bits_per_word)
            df[f'__{tensor} Num Elems'].append(metrics.capacity[(self.hw_level, tensor)])
        df['__Mappings'].append(metrics.state.mapping_of_interest)

        df['Latency'] = metrics.latency
        df['Energy'] = metrics.energy

    def get_result(self):
        return {
            k: pd.DataFrame(v) for k, v in self.compatibility_to_df.items()
        }

    def run(self, rank_shapes):
        n_spatial = len(self.max_spatial)

        # Generate fused_tensor
        unfusable_tensors = {
            t for t in self.tensors if t not in self.fusable_tensors
        }
        for fused_tensors in powerset(self.fusable_tensors):
            fused_tensors = frozenset(fused_tensors)
            unfused_tensors = \
                unfusable_tensors | (self.fusable_tensors - fused_tensors)
            state = SteppedModelState()
            self.model.initialize(state,
                                  0,
                                  self.id_of_einsum_to_eval,
                                  unfused_tensors)

            temporal_ranks_choices = permutations(self.ranks, len(self.ranks))
            spatial_ranks_choices = permutations(self.ranks)
            if reduce(mul, self.max_spatial, 1) == 1:
                spatial_ranks_choices = [[]]
            else:
                spatial_ranks_choices = product(spatial_ranks_choices,
                                                repeat=n_spatial)

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

                        temporal_loops = tuple(zip(temporal_ranks, temporal_tile_shape))
                        if not self.check_mapping(temporal_loops, tile_shape, fused_tensors):
                            continue

                        spatial_loops = [
                            tuple(zip(ranks, spatial_tile_shape))
                            for ranks, spatial_tile_shape
                            in zip(spatial_ranks, spatial_tile_shapes)
                        ]

                        new_state = deepcopy(state)
                        stats = self.partial_model(state=new_state,
                                                   temporal_loops=temporal_loops,
                                                   spatial_loops=spatial_loops,
                                                   retained_tensors=fused_tensors)

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

                                self.store_evaluation_result(
                                    fused_tensors,
                                    temporal_loops,
                                    stats
                                )
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

                            self.store_evaluation_result(
                                fused_tensors,
                                temporal_loops,
                                stats
                            )

    def check_mapping(self, temporal_loops, tile_shape, bypassing):
        if self.mapping_filter is None:
            return True
        return self.mapping_filter(temporal_loops, tile_shape, bypassing)

    def check_stats(self, total_stats):
        if self.stats_filter is None:
            return True
        return self.stats_filter(total_stats)
