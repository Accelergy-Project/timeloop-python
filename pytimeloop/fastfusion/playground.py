from collections import defaultdict
import itertools

import tqdm
from util import fzs

from compatibility import Loop, Compatibility, TensorTiling
from fusionset import InterchangeableSet
from pareto import Pareto
from more_itertools import powerset
from itertools import permutations


def get_compatibility_sets():
    rank_sizes = {
        "M1": 32,
        "K1": 32,
        "N1": 32,
        "N2": 32,
        "N3": 32,
        "N4": 32,
    }
    # fusable_tensors = {"A1", "C1", "C2", "C3", "C4"}
    fusable_tensors = {
        "All"
    }  # {"A1", "C1", "C2", "C3", "C4"} | {"B1", "B2", "B3", "B4"}
    must_fuse = set()  # set(["C1", "C2", "C3", "C4"])

    compatibility_sets = []

    def makecs(einsum_id: str, tensor2rank: dict[str, set[str]]):
        # print(f"Tensor2Rank: {tensor2rank}")

        compatibility_sets = []
        tensors = set(tensor2rank.keys())
        to_test = tensors if "All" in fusable_tensors else fusable_tensors & tensors
        for tn in powerset(to_test):
            # print(f"\t{tn}")
            if (must_fuse & tensors) - set(tn):
                continue
            if tn:
                ranks = set.union(*[tensor2rank[t] for t in tn])
            else:
                ranks = set()
            for r in powerset(ranks):
                # print(f"\t\t{r}")
                perms = list(permutations(r)) if r else [()]
                for perm in perms:
                    # print(f"\t\t\t{perm}")
                    # For every possible prime factorizaton of each rank
                    factors = [1] * len(perm)

                    def make_cs():
                        tiling = {}
                        for t in tn:
                            ranks = tensor2rank[t]
                            loops = [Loop(p, f, False) for p, f in zip(perm, factors)]
                            for i in range(len(loops) - 1, -1, -1):
                                if loops[i].rank_id not in ranks:
                                    loops.pop(i)
                                else:
                                    break
                            tiling[t] = TensorTiling(
                                "GLB" if t in tn else "DRAM", tuple(loops)
                            )
                        for t in tensors - set(tn):
                            tiling[t] = TensorTiling("DRAM", tuple())

                        compatibility_sets.append(
                            Compatibility(einsum_id=einsum_id, tiling=tiling)
                        )

                    if not factors:
                        make_cs()

                    while not all(f == rank_sizes[p] for f, p in zip(factors, perm)):
                        for i in range(len(factors)):
                            factors[i] *= 2
                            if factors[i] > rank_sizes[perm[i]]:
                                factors[i] = 1
                            else:
                                break
                        if any(f == 1 for f in factors):
                            continue
                        make_cs()
        return einsum_id, [
            InterchangeableSet({c}, Pareto.get_dummy()) for c in compatibility_sets
        ]

    compatibility_sets = [
        makecs("M1", {"A1": {"M1", "K1"}, "B1": {"K1", "N1"}, "C1": {"M1", "N1"}}),
        makecs("M2", {"C1": {"M1", "N1"}, "B2": {"N1", "N2"}, "C2": {"M1", "N2"}}),
        makecs("M3", {"C2": {"M1", "N2"}, "B3": {"N2", "N3"}, "C3": {"M1", "N3"}}),
        makecs("M4", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
        makecs("M5", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
        makecs("M6", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
        makecs("M7", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
        makecs("M8", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
        makecs("M9", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
        makecs("M10", {"C3": {"M1", "N3"}, "B4": {"N3", "N4"}, "C4": {"M1", "N4"}}),
    ]
    return compatibility_sets


def run(compatibility_sets):
    einsum, sols = compatibility_sets.pop(0)
    einsums = [einsum]
    first_compatibility = next(iter(sols[0].compatibility))
    seen_tensors = set(first_compatibility.tensors)

    count_prev_buckets = []
    count_next_buckets = []
    count_len_sols = []
    count_len_next_sols = []
    count_len_combined_sols = []
    count_len_combined_next_sols = []
    count_total = []

    while compatibility_sets:
        live_tensors = set.union(set(), *[s[1][0].tensors for s in compatibility_sets])

        # Put together the next set of solutions
        einsum, next_sols = compatibility_sets.pop(0)
        next_live_tensors = set.union(
            set(), *[s[1][0].tensors for s in compatibility_sets]
        )

        first_compatibility = next(iter(next_sols[0].compatibility))
        next_tensors = first_compatibility.tensors

        print("\n\n")
        print("\n\n" + "=" * 100 + f"\nProcessing einsum {einsum}")

        def get_sols_a():
            prev_buckets = InterchangeableSet.bucket(sols, live_tensors)
            next_buckets = InterchangeableSet.bucket(next_sols, seen_tensors)
            # InterchangeableSet.call_on_buckets(
            #     prev_buckets, lambda x: InterchangeableSet.combine_combineable(x)
            # )
            # InterchangeableSet.call_on_buckets(
            #     next_buckets, lambda x: InterchangeableSet.combine_combineable(x)
            # )

            # for k, v in prev_buckets.items():
            #     print(f"Prev Bucket {k}: {len(v)}")
            #     for i in v:
            #         print(f"\t{i}")
            #     prev_len = len(v)
            #     v = InterchangeableSet.combine_combineable(v)
            #     print(f"\t{prev_len}->{len(v)}")
            #     for i in v:
            #         print(f"\t{i}")
            # for k, v in next_buckets.items():
            #     print(f"Next Bucket {k}: {len(v)}")
            print(f"Number of solutions: {len(sols)} x {len(next_sols)}")
            print(f"Number of buckets: {len(prev_buckets)} x {len(next_buckets)}")

            # We can vertical combine the previous buckets now because we know that the choice within
            # each bucket is independent of future choices.
            prev_buckets = InterchangeableSet.combine_combineable(prev_buckets)
            next_buckets = InterchangeableSet.combine_combineable(next_buckets)
            print(
                f"Number of post-combined solutions: {sum(len(v) for v in prev_buckets.values())} x {sum(len(v) for v in next_buckets.values())}"
            )

            new_sols = []
            for s1, s2 in tqdm.tqdm(
                InterchangeableSet.pair_matching_buckets(prev_buckets, next_buckets)
            ):
                new_sols.append(s1.combine(s2))
                # print(f"Combined {s1} with {s2}")

            print(f"A: Generated {len(new_sols)} from {len(sols)} x {len(next_sols)}")
            return new_sols

        def get_sols_b():
            prev_buckets = InterchangeableSet.bucket(sols, live_tensors)
            next_buckets = InterchangeableSet.bucket(next_sols, seen_tensors)
            print(f"Number of solutions: {len(sols)} x {len(next_sols)}")
            print(f"Number of buckets: {len(prev_buckets)} x {len(next_buckets)}")
            InterchangeableSet.call_on_buckets(
                prev_buckets, lambda x: InterchangeableSet.combine_combineable(x)
            )
            InterchangeableSet.call_on_buckets(
                next_buckets, lambda x: InterchangeableSet.combine_combineable(x)
            )
            size_a = sum(len(v) for v in prev_buckets.values())
            size_b = sum(len(v) for v in next_buckets.values())
            print(f"Number of combined solutions: {size_a} x {size_b}")
            prev_tensors = set(fzs(fs.tensors) for fs in prev_buckets)
            next_tensors = set(fzs(fs.tensors) for fs in next_buckets)
            assert len(prev_tensors) == 1

            count_prev_buckets.append(len(prev_buckets))
            count_next_buckets.append(len(next_buckets))
            count_len_sols.append(len(sols))
            count_len_next_sols.append(len(next_sols))
            count_len_combined_sols.append(size_a)
            count_len_combined_next_sols.append(size_b)

            assert len(next_tensors) == 1
            assert prev_tensors == next_tensors
            new_sols = []
            for s1, s2 in tqdm.tqdm(
                InterchangeableSet.pair_matching_buckets_query(
                    prev_buckets, next_buckets
                )
            ):
                new_sols.append(s1.combine(s2))
            count_total.append(len(new_sols))

            print(f"B: Generated {len(new_sols)} from {len(sols)} x {len(next_sols)}")
            if False:
                for s in new_sols:
                    print(f"\t{s}")
                print("\n\n")
                new_sols = InterchangeableSet.combine_combineable(new_sols)
                print(f"Combined {len(new_sols)}")
                for s in new_sols:
                    print(f"\t{s}")

            return new_sols

        def get_sols_c():
            # print(f"Part 1")
            prev_buckets = InterchangeableSet.bucket_multi_level(sols, live_tensors)

            def print_buckets_recursive(buckets, indent=0):
                if isinstance(buckets, dict):
                    for k, v in buckets.items():
                        print(f"{'  ' * indent}{k}: {len(v)}")
                        print_buckets_recursive(v, indent + 1)

            print_buckets_recursive(prev_buckets)

            next_buckets = InterchangeableSet.bucket_multi_level(
                next_sols, seen_tensors | next_live_tensors
            )
            InterchangeableSet.call_on_buckets(
                prev_buckets, InterchangeableSet.combine_combineable
            )
            new_sols = []
            for s1, s2 in InterchangeableSet.pair_matching_buckets(
                prev_buckets, next_buckets
            ):
                new_sols.append(s1.combine(s2))

            print(f"C: Generated {len(new_sols)} from {len(sols)} x {len(next_sols)}")
            return new_sols

        overlap_tensors = next_live_tensors | seen_tensors
        # sols = InterchangeableSet.combine_combineable(sols)
        # next_sols = InterchangeableSet.combine_combineable(next_sols)
        [s.drop_dead(live_tensors) for s in tqdm.tqdm(sols, desc="Dropping Dead 1")]
        [
            s.drop_dead(overlap_tensors)
            for s in tqdm.tqdm(next_sols, desc="Dropping Dead 2")
        ]

        # import timeit

        # sols_a = get_sols_a()
        # sols_b = get_sols_b()
        # sols_a_compatible = set(fzs(s.compatibility) for s in sols_a)
        # sols_b_compatible = set(fzs(s.compatibility) for s in sols_b)
        # for x in sols_a_compatible - sols_b_compatible:
        #     print(f"Missing from B: {x}")
        # assert sols_a_compatible == sols_b_compatible

        # print(f"Time A: {timeit.timeit(get_sols_a, number=1)}")
        # print(f"Time B: {timeit.timeit(get_sols_b, number=1)}")
        # print(f"Time C: {timeit.timeit(get_sols_c, number=1)}")

        sols = get_sols_b()
        print(f"Generated {len(sols)} solutions")
        einsums.append(einsum)

        ops_left = set(s for s, _ in compatibility_sets)
        relevant_tensors = set.union(
            set(), *[s[1][0].tensors for s in compatibility_sets]
        )

        print("\n\n")
        print(f"Relevant Tensors: {relevant_tensors}")

        seen_tensors |= next_tensors

    print(
        f"n_sols,n_next_sols,n_buckets,n_next_buckets,n_combined,n_combined_next,count_total"
    )
    for i in range(1, len(count_prev_buckets)):
        print(
            f"{count_len_sols[i]},{count_len_next_sols[i]},{count_prev_buckets[i]},{count_next_buckets[i]},{count_len_combined_sols[i]},{count_len_combined_next_sols[i]},{count_total[i]}"
        )


if __name__ == "__main__":
    compatibility_sets = get_compatibility_sets()
    run(compatibility_sets)


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
