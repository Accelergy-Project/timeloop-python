from collections import defaultdict
import itertools
from util import fzs

from compatibility import OpCompatibility
from fusionset import FusionSet
from pareto import Pareto


if __name__ == "__main__":
    from more_itertools import powerset
    from itertools import permutations

    rank_sizes = {
        "M1": 8,
        "K1": 8,
        "N1": 8,
        "N2": 8,
        "N3": 8,
        "N4": 8,
    }
    # fusable_tensors = {"A1", "C1", "C2", "C3", "C4"}
    fusable_tensors = {"A1", "C1", "C2", "C3", "C4"} | {"B1", "B2", "B3", "B4"}
    must_fuse = set()  # set(["C1", "C2", "C3", "C4"])

    compatibility_sets = []

    def get_compatibility_sets(
        einsum_id: str, neighbors: set[str], tensor2rank: dict[str, set[str]]
    ):
        # print(f"Tensor2Rank: {tensor2rank}")

        compatibility_sets = []
        tensors = set(tensor2rank.keys())
        all_ranks = set.union(*tensor2rank.values())
        for tn in powerset(tensors & fusable_tensors):
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
                        compatibility_sets.append(
                            OpCompatibility(
                                einsum_id=einsum_id,
                                fused_loops=tuple(
                                    (p, f) for p, f in zip(perm, factors)
                                ),
                                fused_tensors=fzs(tn),
                                ranks=fzs(all_ranks),
                                tensors=fzs(tensors),
                                neighbors=fzs(neighbors),
                            )
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
        return einsum_id, [FusionSet({c}, Pareto({})) for c in compatibility_sets]

    compatibility_sets = [
        get_compatibility_sets(
            "M1",
            {"M2"},
            {
                "A1": {"M1", "K1"},
                "B1": {"K1", "N1"},
                "C1": {"M1", "N1"},
            },
        ),
        get_compatibility_sets(
            "M2",
            {"M1", "M3"},
            {
                "C1": {"M1", "N1"},
                "B2": {"N1", "N2"},
                "C2": {"M1", "N2"},
            },
        ),
        get_compatibility_sets(
            "M3",
            {"M2", "M4"},
            {
                "C2": {"M1", "N2"},
                "B3": {"N2", "N3"},
                "C3": {"M1", "N3"},
            },
        ),
        get_compatibility_sets(
            "M4",
            {"M3"},
            {
                "C3": {"M1", "N3"},
                "B4": {"N3", "N4"},
                "C4": {"M1", "N4"},
            },
        ),
    ]

    einsum, sols = compatibility_sets.pop(0)
    einsums = [einsum]
    first_compatibility = next(iter(sols[0].compatibility))
    seen_einsums = {first_compatibility.einsum_id}
    seen_tensors = set(first_compatibility.tensors)
    seen_ranks = set(first_compatibility.ranks)

    while compatibility_sets:
        # Put together the next set of solutions
        einsum, next_sols = compatibility_sets.pop(0)

        first_compatibility = next(iter(next_sols[0].compatibility))
        next_einsum = {first_compatibility.einsum_id}
        next_tensors = first_compatibility.tensors
        next_ranks = first_compatibility.ranks

        print("\n\n")
        print("\n\n" + "=" * 100 + f"\nProcessing einsum {einsum}")

        unseen_einsums = set(s for s, _ in compatibility_sets) | next_einsum
        unseen_tensors = (
            set.union(set(), *[s[1][0].tensors for s in compatibility_sets])
            | next_tensors
        )
        unseen_ranks = (
            set.union(set(), *[s[1][0].ranks for s in compatibility_sets]) | next_ranks
        )

        def get_sols_a():
            prev_buckets = FusionSet.bucket(
                sols, unseen_einsums, unseen_tensors, unseen_ranks
            )

            next_buckets = FusionSet.bucket(
                next_sols, seen_einsums, seen_tensors, seen_ranks
            )

            print(f"Number of buckets: {len(prev_buckets)} x {len(next_buckets)}")

            # We can vertical combine the previous buckets now because we know that the choice within
            # each bucket is independent of future choices.
            FusionSet.call_on_buckets(
                prev_buckets, lambda x: [FusionSet.vertical_combine(x)]
            )

            new_sols = []
            for s1, s2 in FusionSet.pair_matching_buckets(prev_buckets, next_buckets):
                new_sols.append(s1.combine(s2))

            print(f"A: Generated {len(new_sols)} from {len(sols)} x {len(next_sols)}")
            return new_sols

        def get_sols_b():
            new_sols = [
                s1.combine(s2)
                for s1 in sols
                for s2 in next_sols
                if s1.compatible_with(s2)
            ]
            print(f"B: Generated {len(new_sols)} from {len(sols)} x {len(next_sols)}")
            return new_sols

        def get_sols_c():
            # print(f"Part 1")
            prev_buckets = FusionSet.bucket_multi_level(
                sols, unseen_einsums, unseen_tensors, unseen_ranks
            )

            next_buckets = FusionSet.bucket_multi_level(
                next_sols, seen_einsums, seen_tensors, seen_ranks
            )

            # print(f"Part 2")
            FusionSet.call_on_buckets(prev_buckets, FusionSet.vertical_combine)
            new_sols = []

            for s1, s2 in FusionSet.pair_matching_buckets(prev_buckets, next_buckets):
                new_sols.append(s1.combine(s2))
            import joblib

            # new_sols = joblib.Parallel(n_jobs=64)(
            #     [
            #         joblib.delayed(lambda x1, x2: x1.combine(x2))(x1, x2)
            #         for x1, x2 in FusionSet.pair_matching_buckets(
            #             prev_buckets, next_buckets
            #         )
            #     ]
            # )

            print(f"C: Generated {len(new_sols)} from {len(sols)} x {len(next_sols)}")
            return new_sols

        [s.drop_dead(unseen_einsums) for s in sols]

        # import timeit
        # print(f"Time A: {timeit.timeit(get_sols_a, number=1)}")
        # # print(f"Time B: {timeit.timeit(get_sols_b, number=1)}")
        # print(f"Time C: {timeit.timeit(get_sols_c, number=1)}")

        sols = get_sols_c()
        print(f"Generated {len(sols)} solutions")
        einsums.append(einsum)

        ops_left = set(s for s, _ in compatibility_sets)
        relevant_tensors = set.union(
            set(), *[s[1][0].tensors for s in compatibility_sets]
        )
        relevant_ranks = set.union(set(), *[s[1][0].ranks for s in compatibility_sets])

        print("\n\n")
        print(f"Relevant Tensors: {relevant_tensors}")
        print(f"Relevant Ranks: {relevant_ranks}")

        # bucketed = defaultdict(list)
        # for s in sols:
        #     bucketed[
        #         s.relevant_compatibility(ops_left, relevant_tensors, relevant_ranks)
        #     ].append(s)
        # for k, v in sorted(bucketed.items()):
        #     print(
        #         f"{len(v)} in bucket\n       {'\n       '.join(str(s) for s in k.compatibility)}"
        #     )
        #     for i in sorted(v):
        #         print(f"  {sorted(i.compatibility)}")
        # print("")

        seen_einsums |= next_einsum
        seen_tensors |= next_tensors
        seen_ranks |= next_ranks
