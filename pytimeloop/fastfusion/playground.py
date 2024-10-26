from collections import defaultdict
import itertools

import pandas as pd
import tqdm
from util import fzs

from sim import SIM, Loop, TensorStorage, Tiling
from pareto import Pareto, MAPPING
from more_itertools import powerset
from itertools import permutations


def get_sims():
    RANK_SIZE = 512

    fusable_tensors = {
        "All"
    }  # {"A1", "C1", "C2", "C3", "C4"} | {"B1", "B2", "B3", "B4"}
    fusable_tensors = {"A1", "C1", "C2", "C3", "C4", "C5", "C6", "C7"}
    must_fuse = set()  # set(["C1", "C2", "C3", "C4"])

    sims = []

    def makesim(einsum_id: str, tensor2rank: dict[str, set[str]]):
        sims = []
        tensors = set(tensor2rank.keys())
        to_test = tensors if "All" in fusable_tensors else fusable_tensors & tensors
        for tn in powerset(to_test):
            if (must_fuse & tensors) - set(tn):
                continue
            if tn:
                ranks = set.union(*[tensor2rank[t] for t in tn])
            else:
                ranks = set()
            for r in powerset(ranks):
                perms = list(permutations(r)) if r else [()]
                for perm in perms:
                    factors = [1] * len(perm)

                    def ms():
                        loops = tuple(Loop(p, f, False) for p, f in zip(perm, factors))
                        maxlen = -1
                        tns = []
                        for t in tensors:
                            j = 0
                            for i, l in enumerate(loops):
                                if l.rank_id in tensor2rank[t]:
                                    j = i + 1
                            tns.append(
                                TensorStorage(
                                    tensor_id=t,
                                    backer_id="GLB",
                                    above_loop_index=j if t in tn else 10,
                                    tile_size=1,
                                )
                            )
                            maxlen = max(maxlen, j)
                        m = pd.DataFrame({"Energy": [1], MAPPING: [{}]})
                        sims.append(SIM(Tiling(loops, fzs(tns)), Pareto(m)))

                    if not factors:
                        ms()

                    while not all(f == RANK_SIZE for f, p in zip(factors, perm)):
                        for i in range(len(factors)):
                            factors[i] *= 2
                            if factors[i] > RANK_SIZE:
                                factors[i] = 1
                            else:
                                break
                        if any(f == 1 for f in factors):
                            continue
                        ms()
        return sims

    sims = [
        makesim("M1", {"A1": {"M1", "K1"}, "B1": {"K1", "N1"}, "C1": {"M1", "N1"}}),
        makesim("M2", {"C1": {"M1", "N1"}, "B2": {"N1", "N2"}, "C2": {"M1", "N2"}}),
        makesim("M3", {"C2": {"M1", "N2"}, "B3": {"N2", "N3"}, "C3": {"M1", "N3"}}),
        makesim("M4", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
        makesim("M5", {"C4": {"M1", "N4"}, "B5": {"N4", "N5"}, "C5": {"M1", "N5"}}),
        makesim("M6", {"C5": {"M1", "N5"}, "B6": {"N5", "N6"}, "C6": {"M1", "N6"}}),
        makesim("M7", {"C6": {"M1", "N6"}, "B7": {"N6", "N7"}, "C7": {"M1", "N7"}}),
    ]
    return sims


def run(sims: list[SIM]):
    s: list[SIM] = sims.pop(0)

    count_prev_buckets = []
    count_next_buckets = []
    count_len_sols = []
    count_len_next_sols = []
    count_len_combined_sols = []
    count_len_combined_next_sols = []
    count_total = []

    while sims:
        live_tensors = set.union(set(), *[sim[0].tensor_names for sim in sims])
        ns = sims.pop(0)

        ns = SIM.group_by_left(ns, s[0].tensor_names)
        for s2 in s:
            s2.consolidate(live_tensors)
        s = SIM.group_by_right(SIM.combine_combineable(s, live_tensors), live_tensors)
        for s2 in s.values():
            for s3 in s2:
                assert len(s3.mappings) == len(s3.tilings)

        print("\n\n")
        print("\n\n" + "=" * 100 + f"\n{len(sims) + 1} Remaining\n" + "=" * 100)

        DO_PRINT = False

        combined: list[SIM] = []
        for k in s:
            if k in ns:
                for a, b in itertools.product(s[k], ns[k]):
                    if DO_PRINT:
                        print(f"\t{a.tiling_str()} <--> {b.tiling_str()}")
                    combined.append(a.copy())
                    combined[-1].merge_next(b, set())
                    # combined_keys.append()
            elif DO_PRINT:
                print(f"\tNo match for {s[k][0].tiling_str()}")

        s = combined
        print(f"Generated {len(s)} solutions")

    print(
        f"n_sols,n_next_sols,n_buckets,n_next_buckets,n_combined,n_combined_next,count_total"
    )
    for i in range(1, len(count_prev_buckets)):
        print(
            f"{count_len_sols[i]},{count_len_next_sols[i]},{count_prev_buckets[i]},{count_next_buckets[i]},{count_len_combined_sols[i]},{count_len_combined_next_sols[i]},{count_total[i]}"
        )


if __name__ == "__main__":
    print(f"Getting SIMs")
    sims = get_sims()
    print(f"Running")
    run(sims)


"""


4 Einsums
2048x2048x2048 each

A: 35780 -> 5832 buckets, 5832 total
B: 35780 -> 5832 buckets, 20409 total
AB -> 5832x5832 buckets, 9277 total

AB: 9277 -> 5832 buckets, 8870 total
C: 35780 -> 5832 buckets, 20409 total
ABC -> 5832x5832 buckets, 16838 total

ABC: 16838 -> 5832 buckets, 7296 total
D: 35780 -> 5832 buckets, 5832 total
ABCD: 399 total

"""
