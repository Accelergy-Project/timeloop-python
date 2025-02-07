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
):
    nmappings = []
    nbuckets = []
    
    sims = list(sims.items())
    
    for einsum_id, s in sims:
        print(f'SIM {einsum_id} tensors: {s[0].tensor_names}')

    # TODO: Lookahead by one SIM. If we're going to create a tiling that has loops
    # that are not in the ranks of the next SIM, we should drop that tiling.
    # if pre_filter:
    #     for i in range(len(sims) - 1):
    #         left, right = sims[i], sims[i + 1]
    #         left_live = set.union(set(), *[s[0].tensor_names for s in sims[:i + 1]])
    #         right_live = set.union(set(), *[s[0].tensor_names for s in sims[i + 1:]])
    #         left2, right2 = SIM.get_possibly_compatible(left, right, left_live, right_live)
    #         if not left2 or not right2:
    #             left2, right2 = SIM.get_possibly_compatible(left, right, left_live, right_live)
    #         sims[i], sims[i + 1] = left2, right2
    #         print(f'Filtered {len(left)} -> {len(left2)} SIMs from Einsum {i}')
    #         print(f'Filtered {len(right)} -> {len(right2)} SIMs from Einsum {i + 1}')

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

    n_iterations = 1
    total_iterations = len(sims)
    left_einsum, left = sims.pop(0)
    while sims:
        n_iterations += 1
        nbuckets.append(len(left))
        nmappings.append(sum(len(s.mapping.data) for s in left))

        right_einsum, right = sims.pop(0)
        print(f'\nEinsum {right_einsum} ({n_iterations}/{total_iterations})')
        
        live_tensors = set.union(set(), *[s[0].tensor_names for _, s in sims if s])
        shared_tensors = set(left[0].tensor_names) & set(right[0].tensor_names)

        right_tensors = right[0].tensor_names
        left_tensors = left[0].tensor_names
        
        args = dict(
            left=False,
            live_tensors=live_tensors,
            resource2capacity=resource2capacity,
            shared_tensors=shared_tensors,
        )

        left = sorted(left, key=lambda x: len(x.mapping.data), reverse=True)
        right = sorted(right, key=lambda x: len(x.mapping.data), reverse=True)
        lr = parallel(
            [delayed(lambda l: l.left_consolidate(live_tensors, resource2capacity, shared_tensors))(l) for l in left] + 
            [delayed(lambda l: l.consolidate(live_tensors, resource2capacity, shared_tensors))(l) for l in right],
            pbar=f"Consolidating {left_einsum} <--> {right_einsum}",
        )
        left, right = lr[:len(left)], lr[len(left):]
        print_time(f"Consolidating")
        
        # left = [t for t in left if not t.tiling.tags]
            
        # if not sims:
        #     left2 = SIM.combine_combineable(left, live_tensors | right_tensors)
        #     # left2 = left
        #     print(f'Left length: {len(left)} -> {len(left2)}')
        #     for l in left:
        #         print(f'\t{l.tiling}')
        #     for l2 in left2:
        #         print(f'\t-> {l2.tiling}')
        #     left = left2
            
        # if not sims:
        #     right2 = [t for t in right if not t.tiling.tags]
        #     for t in SIM.combine_combineable(right, live_tensors | left_tensors):
        #         print(f'Right: {t.tiling}')
        #     for t in SIM.combine_combineable(right2, live_tensors | left_tensors):
        #         print(f'Right2: {t.tiling}')
        #     right = SIM.combine_combineable(right, live_tensors | left_tensors)
        #     right2 = SIM.combine_combineable(right2, live_tensors | left_tensors)
        #     right_grouped = SIM.group_right(right, left_tensors, drop_tags=True)
        #     right2_grouped = SIM.group_right(right2, left_tensors, drop_tags=True)
        #     left2 = SIM.group_left(left, right_tensors, drop_tags=True)
        #     from pytimeloop.fastfusion.sim import Tiling, TensorStorage, Tags, fzs
        #     t = Tiling((), fzs((TensorStorage('Filter3', 0, 0, 16), TensorStorage('Fmap3', 0, 0, 4), TensorStorage('Fmap4', 0, 0, 16))), Tags(fzs()))
        #     for k in left2:
        #         if k in right_grouped and k not in right2_grouped:
        #             print(f'Right {k} not in right2')
        #         if k in right2_grouped and k not in right_grouped:
        #             print(f'Right2 {k} not in right')
        #         # if k in right_grouped and k in right2_grouped:
        #         #     for a, b in itertools.product(left[k], right_grouped[k]):
        #         #         if a.tiling != b.tiling:
        #         #             print(f'Right {k} not equal
                
        left = SIM.combine_combineable(left, live_tensors | right_tensors)
        right = SIM.combine_combineable(right, live_tensors | left_tensors)
        print_time(f"Combining")

        # Group left and right into buckets
        right = SIM.group_right(right, left_tensors, drop_tags=True)
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
                        new = a.merge_next(b, live_tensors, delay=DELAY_MERGE)
                        # This "if" check is only for speed. If we can't merge this with
                        # anything later on, we don't need to add it to the list.
                        if get_possible_translations(new.tiling) or not sims:
                            combined.append(new)
                        if DO_PRINT:
                            s = f"\t{a.tiling} <--> {b.tiling}"
                            s += f" --> {combined[-1].tiling}"
                            s += f"({len(a.mapping.data)})x({len(b.mapping.data)})"
                            print(s)
            if DO_PRINT and not found:
                for a in left[k]:
                    print(f"\tNo match for {a.tiling}")

        print_time("Bucket merging")
        
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
