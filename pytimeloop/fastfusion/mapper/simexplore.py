from collections import defaultdict
from collections.abc import Mapping
import copy
import itertools
import time

import pandas as pd
from joblib import delayed
from tqdm import tqdm

from pytimeloop.fastfusion.sim import SIM
from pytimeloop.fastfusion.pareto import Pareto, check_correctness
from pytimeloop.fastfusion.util import parallel, debugger_active


def explore_fusion(
    einsum_to_result: Mapping,
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
):
    return fuse_sims(
        mapping2sims(einsum_to_result), resource2capacity, return_nmappings_nbuckets
    )


def mapping2sims(einsum_to_result: Mapping):
    r = {}
    for einsum_id, compat_dict in einsum_to_result.items():
        r[einsum_id] = [paretofy(k, v) for k, v in compat_dict.items()]
    return list(r.values())


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
    sims: list[SIM],
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
    pre_filter: bool = True
):
    nmappings = []
    nbuckets = []
    resource2capacity = resource2capacity or {}
    sims = [s for s in sims]
    
    for i, s in enumerate(sims):
        print(f'SIM {i} tensors: {s[0].tensor_names}')
        
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

    left = sims.pop(0)

    init_print_time()
    
    if not sims:
        sims = copy.deepcopy(sims)
        left = consolidate(
            x=left,
            left=True,
            live_tensors=set(),
            resource2capacity=resource2capacity,
            shared_tensors=set(),
        )
        
    # TODO: Lookahead by one SIM. If we're going to create a tiling that has loops
    # that are not in the ranks of the next SIM, we should drop that tiling.

    while sims:
        nbuckets.append(len(left))
        nmappings.append(sum(len(s.mapping.data) for s in left))

        right = sims.pop(0)
        live_tensors = set.union(set(), *[s[0].tensor_names for s in sims if s])
        shared_tensors = set(left[0].tensor_names) & set(right[0].tensor_names)

        right_tensors = right[0].tensor_names
        left_tensors = left[0].tensor_names
        
        args = dict(
            left=False,
            live_tensors=live_tensors,
            resource2capacity=resource2capacity,
            shared_tensors=shared_tensors,
        )

        # right = consolidate(right, left=False, **args)
        # left = consolidate(left, left=True, **args)

        left = SIM.combine_combineable(left, live_tensors | right_tensors)
        right = SIM.combine_combineable(right, live_tensors | left_tensors)

        left = sorted(left, key=lambda x: len(x.mapping.data), reverse=True)
        right = sorted(right, key=lambda x: len(x.mapping.data), reverse=True)

        left = parallel([delayed(lambda l: l.left_consolidate(live_tensors, resource2capacity, shared_tensors))(l) for l in left], pbar="Left consolidate")
        right = parallel([delayed(lambda l: l.consolidate(live_tensors, resource2capacity, shared_tensors))(l) for l in right], pbar="Right consolidate")

        # Group left and right into buckets
        right = SIM.group_right(right, left_tensors)
        left = SIM.group_left(left, right_tensors)
        print_time("Grouping")

        for v in list(left.values()) + list(right.values()):
            for s in v:
                for t in list(s.tensors):
                    if t not in live_tensors:
                        del s.tensors[t]

        # left = {k: SIM.combine_combineable(v, live_tensors | right_tensors) for k, v in left.items()}
        # right = {k: SIM.combine_combineable(v, live_tensors | left_tensors) for k, v in right.items()}

        print_time("Consolidating")

        DO_PRINT = False
        DELAY_MERGE = not debugger_active()

        combined: list[SIM] = []
        for k in left:
            if k in right:
                for a, b in itertools.product(left[k], right[k]):
                    a: SIM
                    b: SIM
                    combined.append(a.merge_next(b, live_tensors, delay=DELAY_MERGE))
                    combined[-1]._predicted_mappings = len(a.mapping.data) * len(b.mapping.data)
                    if DO_PRINT:
                        s = f"\t{a.tiling} <--> {b.tiling}"
                        s += f" --> {combined[-1].tiling}"
                        s += f"({len(a.mapping.data)})x({len(b.mapping.data)})"
                        print(s)
            elif DO_PRINT:
                for a in left[k]:
                    print(f"\tNo match for {a.tiling}")

        if not combined:
            print(f'No valid combinations found.')

        print_time("Bucket merging")
        
        if DELAY_MERGE:
            combined = sorted(combined, key=lambda x: x._predicted_mappings, reverse=True)
            for c, mapping in zip(combined, parallel([c.mapping for c in combined], pbar='Merging mappings')):
                c.mapping = mapping

        print_time("Mapping merging")

        print(
            f"\tCombining {sum(len(s) for s in left)}({len(left)}) x {sum(len(s) for s in right)}({len(right)}) -> {len(combined)}"
        )
        # if DO_PRINT:
        #     for k in right:
        #         if k not in left:
        #             for b in right[k]:
        #                 print(f"\tREVERSE: No match for {b.tiling}")

        left = combined
        print(f"Number of buckets: {len(left)}")
        n_mappings = sum(len(s.mapping.data) for s in left)
        print(f"Number of mappings: {n_mappings}")
        print(f"Mappings per bucket: {n_mappings / len(left)}")

    for s in left:
        s.left_consolidate(set(), resource2capacity)
    s_final = SIM.combine_combineable(left, set())[0]
    data = s_final.mapping.data
    # check_correctness(data, set())

    print_total_time()

    if return_nmappings_nbuckets:
        return data, nmappings, nbuckets
    return data


def paretofy(k, v):
    return SIM(k, Pareto(pd.DataFrame(v).fillna(0)))
