from collections import defaultdict
from collections.abc import Mapping
import copy
import itertools
import time

import pandas as pd
from joblib import delayed

from pytimeloop.looptree.equivalent_ranks import PairwiseEquivalentRanks

from pytimeloop.fastfusion.sim import SIM, Loop, Tiling
from pytimeloop.fastfusion.pareto import Pareto
from pytimeloop.fastfusion.util import fzs, parallel, debugger_active


def explore_fusion(
    einsum_to_result: Mapping,
    equivalent_ranks: PairwiseEquivalentRanks,
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
):
    return fuse_sims(
        mapping2sims(einsum_to_result),
        equivalent_ranks,
        resource2capacity,
        return_nmappings_nbuckets
    )


def mapping2sims(einsum_to_result: Mapping):
    r = {}
    for einsum_id, compat_dict in einsum_to_result.items():
        r[einsum_id] = [paretofy(k, v) for k, v in compat_dict.items()]
    return list(r.values())
def paretofy(k, v):
    return SIM(k, Pareto(pd.DataFrame(v).fillna(0)))


prev_time = 0
total_time = defaultdict(int)


def init_print_time():
    global prev_time, total_time
    prev_time = time.time()
    total_time = defaultdict(int)


def print_time(what: str):
    global prev_time
    t = time.time() - prev_time
    print(f"{what}: {t}")
    total_time[what] += t
    prev_time = time.time()


def print_total_time():
    print(f"\n======== Total time ========")
    for k, v in total_time.items():
        print(f"{k}: {v}")
    print(f"============================\n")


def consolidate(
    x,
    left: bool,
    live_tensors: set,
    resource2capacity: dict,
    shared_tensors: set,
):
    # taret = "left_consolidate" if left else "consolidate"
    # pbar = "Left consolidate" if left else "Right consolidate"
    # x = parallel(
    #     [delayed(getattr(x2, taret))(live_tensors, resource2capacity, shared_tensors) for x2 in x],
    #     pbar=pbar
    # )
    
    # for x2 in x:
    #     if left:
    #         x2.left_consolidate(live_tensors, resource2capacity, shared_tensors)
    #     else:
    #         x2.consolidate(live_tensors, resource2capacity, shared_tensors)
    x = SIM.combine_combineable(x, live_tensors)
    # We freed these
    for x2 in x:
        for t in list(x2.tensors):
            if t not in live_tensors:
                del x2.tensors[t]
    return x


def fuse_sims(
    sims: dict[str, list[SIM]],
    pairwise_equivalent_ranks: PairwiseEquivalentRanks,
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
    lookahead_filter: bool = True,
):
    nmappings = []
    nbuckets = []
    
    sims = list(sims.items())
    
    for einsum_id, s in sims:
        print(f'SIM {einsum_id} tensors: {s[0].tensor_names}')
    init_print_time()
    if len(sims) == 1:
        left = copy.deepcopy(sims[0][1])
        sims = []
        left = consolidate(
            x=left,
            left=True,
            live_tensors=set(),
            resource2capacity=resource2capacity,
            shared_tensors=set(),
        )

    class SIMHolder:
        def __init__(self, einsum_id: str, sim_list: list[SIM]):
            self.einsum_id: str = einsum_id
            self.sims: list[SIM] = sim_list
            self.tensor_names: set[str] = set(sim_list[0].tensor_names)
            
        def __getitem__(self, i):
            return self.sims[i]
            
    sims = [SIMHolder(*s) for s in sims]
    
    # Right consolidate all SIMs
    for i, sim_holder in enumerate(sims):
        if i == 0:
            continue
        left_tensors = set.union(set(), *[s.tensor_names for s in sims[:i]])
        right_tensors = set.union(set(), *[s.tensor_names for s in sims[i+1:]])
        live_tensors = right_tensors
        shared_tensors = left_tensors & sim_holder.tensor_names
        sim_holder.sims = sorted(sim_holder.sims, key=lambda x: len(x.mapping.data), reverse=True)
        sim_holder.sims = parallel(
            [delayed(lambda x: x.consolidate(live_tensors, resource2capacity, shared_tensors))(x) for x in sim_holder.sims],
            pbar=f"Right consolidate {sim_holder.einsum_id}"
        )
        sim_holder.sims = SIM.combine_combineable(sim_holder.sims, left_tensors | right_tensors)
        if i > 0:
            sim_holder.sims = SIM.group_right(sim_holder.sims, left_tensors, drop_tags=True)
    print(f'Initial consolidate')
    
    n_iterations = 1
    total_iterations = len(sims)
    left_holder = sims.pop(0)
    left, left_einsum, left_tensors = left_holder.sims, left_holder.einsum_id, left_holder.tensor_names
    while sims:
        n_iterations += 1
        nbuckets.append(len(left))
        nmappings.append(sum(len(s.mapping.data) for s in left))

        right_holder = sims.pop(0)
        right, right_einsum, right_tensors = right_holder.sims, right_holder.einsum_id, right_holder.tensor_names
        print(f'\nEinsum {right_einsum} ({n_iterations}/{total_iterations})')
        
        live_tensors = set.union(set(), *[s.tensor_names for s in sims])
        shared_tensors = set(left_tensors) & set(right_tensors)

        left = parallel(
            [delayed(lambda l: l.left_consolidate(live_tensors, resource2capacity, shared_tensors))(l) for l in left],
            pbar=f"Left consolidate {left_einsum}"
        )
        print_time(f"Consolidating")
        
        left = SIM.combine_combineable(left, live_tensors | right_tensors)
        print_time(f"Combining")

        # Group left and right into buckets
        left = SIM.group_left(left, right_tensors, drop_tags=True)
        print_time("Grouping")

        for v in list(left.values()) + list(right.values()):
            for s in v:
                for t in list(s.tensors):
                    if t not in live_tensors:
                        del s.tensors[t]

        def get_possible_translations(t: Tiling):
            def translate_loop(l: Loop):
                compatible_ranks = set.intersection(
                    *(pairwise_equivalent_ranks[n] for n in l.rank_names)
                )
                for n in compatible_ranks:
                    yield Loop(fzs((n,)), l.bound, l.is_spatial)
            for loops in itertools.product(*map(translate_loop, t.loops)):
                yield Tiling(loops, t.tensors, t.tags)

        DO_PRINT = False
        DELAY_MERGE = not debugger_active()

        combined: list[SIM] = []
        for k in left:
            found = False
            for k_translated in get_possible_translations(k):
                for a, b in itertools.product(left[k], right.get(k_translated, [])):
                    a: SIM
                    b: SIM
                    if a.tiling.tags.are_compatible_with(b.tiling.tags):
                        found = True
                        combined.append(a.merge_next(b, live_tensors, delay=DELAY_MERGE))
                        if DO_PRINT:
                            s = f"\t{a.tiling} <--> {b.tiling}"
                            s += f" --> {combined[-1].tiling}"
                            s += f"({len(a.mapping.data)})x({len(b.mapping.data)})"
                            print(s)
            if DO_PRINT and not found:
                for a in left[k]:
                    print(f"\tNo match for {a.tiling}")

        print_time("Bucket merging")

        # This check is only for speed. If we can't merge with
        # anything later on, we don't need to add it to the list.
        if sims and lookahead_filter:
            prev_len = len(combined)
            next_right_tensors = sims[0].tensor_names
            combined = SIM.group_left(combined, next_right_tensors, drop_tags=True)
            for k in list(combined):
                if not any(kt in sims[0].sims for kt in get_possible_translations(k)):
                    del combined[k]
                    # print(f'\tNo match for {k}')
            combined = list(itertools.chain.from_iterable(combined.values()))
            print(f'Pruned {prev_len - len(combined)}/{prev_len} ({len(combined)/prev_len*100:.2f}% remaining)')
            print_time("Removing mappings that can't be combined later.")
        
        if DELAY_MERGE:
            combined = sorted(combined, key=lambda x: x.n_pre_prune_mappings, reverse=True)
            for c, mapping in zip(combined, parallel([c.mapping for c in combined], pbar=f'Merging mappings {left_einsum} <--> {right_einsum}')):
                c.mapping = mapping

        print_time("Mapping merging")

        print(f"\tCombining {sum(len(s) for s in left)}({len(left)}) x {sum(len(s) for s in right)}({len(right)}) -> {len(combined)}")
        if DO_PRINT:
            for k in right:
                if k not in left:
                    for b in right[k]:
                        print(f"\tREVERSE: No match for {b.tiling}")

        left = combined
        left_einsum = right_einsum
        left_tensors |= right_tensors
        print(f"\tNumber of buckets for Einsum {left_einsum}: {len(left)}")
        n_mappings = sum(len(s.mapping.data) for s in left)
        print(f"\tNumber of mappings for Einsum {left_einsum}: {n_mappings}")
        print(f"\tMappings per bucket for Einsum {left_einsum}: {n_mappings / len(left)}")

    for s in left:
        s.left_consolidate(None, resource2capacity)
    s_final = SIM.combine_combineable(left, set(), drop_tags=True)
    assert len(s_final) == 1
    data = s_final[0].mapping.data
    # check_correctness(data, set())

    print_total_time()

    if return_nmappings_nbuckets:
        return data, nmappings, nbuckets
    return data
