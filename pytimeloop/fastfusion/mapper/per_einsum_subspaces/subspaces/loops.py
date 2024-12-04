from collections.abc import Callable
from itertools import permutations


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
                       logfunc: Callable=None):
    original = mapping.copy()

    for r in range(len(ranks) + 1):
        for ordered_ranks in permutations(ranks, r=r):
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