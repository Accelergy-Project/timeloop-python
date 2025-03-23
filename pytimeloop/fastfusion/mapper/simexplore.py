from collections import defaultdict
from collections.abc import Mapping
import copy
import itertools
import time

import pandas as pd
from joblib import delayed

from pytimeloop.looptree.equivalent_ranks import PairwiseEquivalentRanks

from pytimeloop.fastfusion.sim import SIM, Loop, Tiling
from pytimeloop.fastfusion.pareto import VALID, Pareto
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
        return_nmappings_nbuckets,
    )


def mapping2sims(einsum_to_result: Mapping):
    r = {}
    for einsum_id, compat_dict in einsum_to_result.items():
        r[einsum_id] = [paretofy(k, v) for k, v in compat_dict.items()]
    return list(r.values())


def paretofy(k, v):
    return SIM(k, Pareto(pd.DataFrame(v).fillna(0)))


def get_possible_translations(
    t: Tiling, 
    pairwise_equivalent_ranks: dict[str, set[str]],
    full_equivalent_ranks: dict[str, set[str]],
    right_ranks: set[str]
):
    # Fused ranks should be transitive, but if a fused loop indexes into two
    # different ranks in the next Einsum, we can't fuse becuase it will tile in
    # multiple directions.
    #
    # The first union checks what loops we CAN fuse with in the next Einsum. The
    # second union checks what loops MUST index into in the next
    #
    # Einsum. If we alias into multiple ranks, we can't fuse. Otherwise, try out
    # each possible rank.
    def translate_loop(l: Loop):
        compatible_ranks = set.union(
            *(full_equivalent_ranks[n] for n in l.rank_names)
        ) & right_ranks
        pairwise_compatible_ranks = set.union(
            *(pairwise_equivalent_ranks[n] for n in l.rank_names)
        ) & right_ranks
        if len(pairwise_compatible_ranks) > 1:
            return
        for n in compatible_ranks:
            yield Loop(fzs((n,)), l.bound, l.is_spatial)

    for loops in itertools.product(*map(translate_loop, t.loops)):
        yield Tiling(loops, t.storage, t.tags)

prev_time = 0
total_time = defaultdict(int)


def init_print_time():
    global prev_time, total_time
    prev_time = time.time()
    total_time = defaultdict(int)


def print_time(what: str):
    global prev_time
    t = time.time() - prev_time
    print(f"{what}: {t:.2f} seconds")
    total_time[what] += t
    prev_time = time.time()


def print_total_time():
    print(f"\n======== Total time ========")
    for k, v in total_time.items():
        print(f"{k}: {v:.2f} seconds")
    total = sum(total_time.values())
    if total > 60:
        print(f"\nTotal: {total:.2f} seconds ({total/60:.2f} minutes)")
    else:
        print(f"\nTotal: {total:.2f} seconds")
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
        for t in list(x2.storage):
            if t not in live_tensors:
                del x2.storage[t]
    return x


class GroupOfSIMsHolder:
    def __init__(self, einsum_id: str, sim_list: list[SIM]):
        self.einsum_id: str = einsum_id
        self.sims: list[SIM] = sim_list
        self.tensor_names: set[str] = set(sim_list[0].tensor_names)

    def __getitem__(self, i):
        return self.sims[i]


def fuse_sims(
    sims: dict[str, list[SIM]],
    pairwise_equivalent_ranks: PairwiseEquivalentRanks,
    einsum2ranks: dict[str, set[str]],
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
    lookahead_filter: bool = True,
    optimus_fused_group_constraint: bool=False,
    optimus_optimizations_only: bool=False,
):
    full_equivalent_ranks = {k: set(v) for k, v in pairwise_equivalent_ranks.items()}
    changed = True
    while changed:
        changed = False
        for r in full_equivalent_ranks:
            for r2 in list(full_equivalent_ranks[r]):
                for r3 in list(full_equivalent_ranks[r2]):
                    if r3 in full_equivalent_ranks[r]:
                        continue
                    changed = True
                    full_equivalent_ranks[r].add(r3)

    for sim_list in sims.values():
        for s in sim_list:
            if VALID in s.mapping.data:
                s.mapping.data = s.mapping.data[s.mapping.data[VALID] == 1]
    
    nmappings = []
    nbuckets = []
    
    n_evaluations = 0

    sims = list(sims.items())
    n_sims = len(sims)

    for einsum_id, s in sims:
        print(f"SIM {einsum_id} tensors: {s[0].tensor_names}")
    init_print_time()
    if len(sims) == 1:
        left = consolidate(
            x=copy.deepcopy(sims[0][1]),
            left=True,
            live_tensors=set(),
            resource2capacity=resource2capacity,
            shared_tensors=set(),
        )
        sims = []

    sims = [GroupOfSIMsHolder(*s) for s in sims]

    # ======================================================================
    # Initial consolidate and group all SIMs
    # ======================================================================
    for i, sim_holder in enumerate(sims):
        if i == 0:
            continue
        left_tensors = set.union(set(), *[s.tensor_names for s in sims[:i]])
        right_tensors = set.union(set(), *[s.tensor_names for s in sims[i + 1 :]])
        live_tensors = right_tensors
        shared_tensors = left_tensors & sim_holder.tensor_names
        sim_holder.sims = sorted(
            sim_holder.sims, key=lambda x: len(x.mapping.data), reverse=True
        )
        sim_holder.sims = SIM.right_consolidate(
            sim_holder.sims,
            live_tensors,
            resource2capacity,
            shared_tensors,
            pbar=f"Inital consolidate {sim_holder.einsum_id}",
        )
        sim_holder.sims = SIM.combine_combineable(
            sim_holder.sims, left_tensors | right_tensors
        )
        if i > 0:
            sim_holder.sims = SIM.group_right(
                sim_holder.sims, left_tensors, drop_tags=True
            )
    print_time(f"Initial consolidate and group")

    n_iterations = 0
    total_iterations = len(sims)

    def grab_sim_holder() -> tuple[dict[Tiling, list[SIM]], str, set[str]]:
        nonlocal n_iterations
        n_iterations += 1
        holder = sims.pop(0)
        return holder.sims, holder.einsum_id, holder.tensor_names

    if sims:
        left, left_einsum, left_tensors = grab_sim_holder()

    partial_mapping_size = 1
    while sims:
        # ======================================================================
        # Grab new Einsum from the right. Record logging data and find still
        # tensors that will be live after this Einsum.
        # ======================================================================
        nbuckets.append(len(left))
        nmappings.append(sum(len(s.mapping.data) for s in left))
        right, right_einsum, right_tensors = grab_sim_holder()
        right_ranks = einsum2ranks[right_einsum]
        print(f"\nEinsum {right_einsum} ({n_iterations}/{total_iterations})")
        
        partial_mapping_size += 1

        live_tensors = set.union(set(), *[s.tensor_names for s in sims])
        shared_tensors = set(left_tensors) & set(right_tensors)

        # ======================================================================
        # Clean up the previously-combined SIMs. Consolidate, combine, group
        # them into buckets.
        # ======================================================================
        left = SIM.left_consolidate(
            left,
            live_tensors,
            resource2capacity,
            shared_tensors,
            pbar=f"Consolidate {left_einsum}",
        )
        if optimus_fused_group_constraint:
            left = [s for s in left if len(set(l.memory_name for l in s.tiling.storage if l.tensor_name in live_tensors)) <= 1]

        print_time(f"Consolidating")
        
        # Optimus can't combine SIMs that have reserved data
        left = SIM.combine_combineable(left, live_tensors | right_tensors, combine_reservations=not optimus_optimizations_only)

        print_time(f"Combining")
        # Group left and right into buckets
        left = SIM.group_left(left, right_tensors, drop_tags=True)
        print_time("Grouping")

        # ======================================================================
        # Remove dead tensors from left and right. This happens after grouping
        # because we only reserve space for shared tensors after it's dead. This
        # is in case the tensor lifetime extends beyond the Einsums for which it
        # is used.
        # ======================================================================
        SIM.remove_dead_tensors(
            [s for lr in [left, right] for v in lr.values() for s in v], live_tensors
        )

        DO_PRINT = False
        DELAY = not debugger_active()

        # Optimus doesn't do compatibility-based bagging. Compares every mapping to every other mapping.
        if optimus_optimizations_only:
            n_left_mappings = sum(len(s.mapping.data) for k in left.values() for s in k)
            n_right_mappings = sum(len(s.mapping.data) for k in right.values() for s in k)
            n_evaluations += partial_mapping_size * n_left_mappings * n_right_mappings

        # ======================================================================
        # Merge the left and right buckets.
        # ======================================================================
        combined: list[SIM] = []
        for k in left:
            found = False
            for k_translated in get_possible_translations(
                k, pairwise_equivalent_ranks, full_equivalent_ranks, right_ranks
            ):
                for a, b in itertools.product(left[k], right.get(k_translated, [])):
                    if a.tiling.tags.are_compatible_with(b.tiling.tags):
                        found = True
                        combined.append(a.merge_next(b, live_tensors, delay=DELAY))
                        if not DELAY and not optimus_optimizations_only:
                            n_evaluations += len(a.mapping.data) * len(b.mapping.data)
                        if DO_PRINT:
                            s = f"\t{a.tiling} <--> {b.tiling}"
                            s += f" --> {combined[-1].tiling}"
                            s += f"({len(a.mapping.data)})x({len(b.mapping.data)})"
                            print(s)
            if DO_PRINT and not found:
                for a in left[k]:
                    print(f"\tNo match for {a.tiling}")

        if DO_PRINT:
            for k in right:
                if k not in left:
                    for b in right[k]:
                        print(f"\tREVERSE: No match for {b.tiling}")

        print_time("Bucket merging")

        # ======================================================================
        # Look ahead to the next Einsum and see if any of our buckets will not
        # be able to merge with it. If so, we can drop them immediately.
        # ======================================================================
        # Optimus can't look ahead to future Einsums to see if we'll have
        # compatibilty problems
        if sims and lookahead_filter and not optimus_optimizations_only:
            prev_len = len(combined)
            next_right_tensors = sims[0].tensor_names
            next_right_ranks = einsum2ranks[sims[0].einsum_id]
            combined = SIM.group_left(combined, next_right_tensors, drop_tags=True)
            for k in list(combined):
                translations = get_possible_translations(
                    k, pairwise_equivalent_ranks, full_equivalent_ranks, next_right_ranks
                )
                if not any(kt in sims[0].sims for kt in translations):
                    list(get_possible_translations(
                        k, pairwise_equivalent_ranks, full_equivalent_ranks, next_right_ranks
                    ))
                    if DO_PRINT:
                        for b in combined[k]:
                            print(f'\tLOOKAHEAD: No match for {b.tiling}')
                    del combined[k]
            if not combined:
                raise ValueError("No match found for any bucket")
            combined = list(itertools.chain.from_iterable(combined.values()))
            print(
                f"Removed {prev_len - len(combined)}/{prev_len} ({len(combined)/prev_len*100:.2f}% remaining)"
            )
            print_time("Removing mappings that can't be combined later")

        if not combined:
            raise ValueError("No match found for any bucket")

        # ======================================================================
        # If we delayed the mapping merging, do it now.
        # ======================================================================
        if DELAY:
            mappings = parallel(
                [c.mapping for c in combined],
                pbar=f"Merging mappings {left_einsum} <--> {right_einsum}",
                return_as="generator",
            )
            for c, mapping in zip(combined, mappings):
                c.mapping = mapping
                n_evaluations += len(mapping.data)
        print_time("Mapping merging")

        # ======================================================================
        # Print statements
        # ======================================================================
        print(
            f"\tCombining {sum(len(s) for s in left)}({len(left)}) x {sum(len(s) for s in right)}({len(right)}) -> {len(combined)}"
        )

        n_mappings = sum(len(s.mapping.data) for s in combined)
        for_einsum_text = f"for Einsum {right_einsum}"
        print(f"\tNumber of buckets {for_einsum_text}: {len(combined)}")
        print(f"\tNumber of mappings {for_einsum_text}: {n_mappings}")
        print(f"\tMappings per bucket {for_einsum_text}: {n_mappings / len(combined)}")

        # ======================================================================
        # Update left for the next iteration.
        # =================================================================
        left = combined
        left_einsum = right_einsum
        left_tensors |= right_tensors

    # ======================================================================
    # Final consolidate and group
    # ======================================================================
    left = SIM.left_consolidate(left, None, resource2capacity, pbar="Final consolidate")
    s_final = SIM.combine_combineable(left, set(), drop_tags=True)
    assert len(s_final) == 1
    data = s_final[0].mapping.data
    # check_correctness(data, set())

    print_total_time()

    if return_nmappings_nbuckets:
        return data, nmappings, nbuckets
    return data, (n_evaluations, 0)
