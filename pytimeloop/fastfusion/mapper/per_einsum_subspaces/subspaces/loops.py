from collections.abc import Callable
from itertools import permutations
from pytimeloop.fastfusion.mapper.constraints import SEPARATOR, WILDCARD


def make_spatial_fors(mapping,
                      ranks,
                      max_factor):
    original = mapping.copy()

    for r in range(len(ranks) + 1):
        for ordered_ranks in permutations(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_spatial(
                    r, factor_constraint=f"<={max_factor}"
                )
            yield mapping


def make_temporal_fors(mapping,
                       ranks,
                       dataflow_constraint: list=None,
                       logfunc: Callable=None):
    if dataflow_constraint is None:
        top_ranks = []
        other_ranks = ranks
    else:
        if SEPARATOR in dataflow_constraint:
            i = dataflow_constraint.index(SEPARATOR)
            disallowed_ranks = set(dataflow_constraint[i+1:])
            dataflow_constraint = dataflow_constraint[:i]
        else:
            disallowed_ranks = set()

        num_wildcards = sum(map(lambda x: x is WILDCARD, dataflow_constraint))
        top_ranks = []
        if num_wildcards == 0:
            top_ranks = dataflow_constraint
            other_ranks = set()
        elif num_wildcards == 1:
            top_ranks = dataflow_constraint[:-1]
            other_ranks = set(ranks) - set(top_ranks) - disallowed_ranks
        else:
            raise NotImplementedError("Constraint not implemented")

    original = mapping.copy()
    for r in top_ranks:
        original.add_temporal(r)

    for r in range(len(other_ranks) + 1):
        for ordered_ranks in permutations(other_ranks, r=r):
            mapping = original.copy()
            if logfunc is not None:
                logfunc(f"{ordered_ranks}")
            for r in ordered_ranks:
                mapping.add_temporal(r)
            yield mapping


def make_temporal_fors_with_smallest_tile(original, ranks):
    for ordered_ranks in permutations(ranks):
        mapping = original.copy()
        for r in ordered_ranks:
            mapping.add_temporal(r, tile_shape=1)
        yield mapping


def make_temporal_fors_in_order(original, ranks):
    for i in range(len(ranks)+1):
        mapping = original.copy()
        for r in ranks[:i]:
            mapping.add_temporal(r)
        yield mapping

