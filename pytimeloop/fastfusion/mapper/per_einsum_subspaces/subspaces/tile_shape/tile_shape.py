from collections import defaultdict
from functools import reduce
from operator import mul

from .shape_subspace import ShapeSubspace

from pytimeloop.looptree.des import LooptreeOutput


def explore_tile_shape(
    mapping,
    rank_shapes,
    compiled_result,
    max_capacity,
    max_fanout,
    only_count=False
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
    num_valid_tile_shapes = 0

    shape_subspace = iter(ShapeSubspace(
            rank_shapes,
            ranks,
            tile_constraints=tile_constraints,
            factor_constraints=factor_constraints,
    ))
    yield shape_subspace
    for shape in shape_subspace:
        num_tile_shapes += 1
        if only_count:
            continue

        result = LooptreeOutput()
        result.ops = call_with_arg(compiled_result.ops, shape)
        result.temporal_steps = call_with_arg(compiled_result.temporal_steps, shape)
        result.fanout = call_with_arg(compiled_result.fanout, shape)
        result.occupancy = call_with_arg(compiled_result.occupancy, shape)
        result.fills = call_with_arg(compiled_result.fills, shape)
        result.reads_to_parent = call_with_arg(compiled_result.reads_to_parent, shape)

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

        if not invalid_spatial:
            num_valid_tile_shapes += 1
            yield shape, result
            
    return num_tile_shapes, num_valid_tile_shapes


def call_with_arg(f, arg):
    if isinstance(next(iter(f.values())), tuple):
        return {k: (v[0], v[1](*arg)) for k, v in f.items()}
    else:
        return {k: v(*arg) for k, v in f.items()}

