from collections import deque
from io import TextIOBase

from .fusionset import Cascade


def explore_fusion_sets(
    compatibility_set_queue: deque, verbose_stream: TextIOBase = None
):
    einsum, sols = compatibility_set_queue.popleft()
    einsums = [einsum]
    first_compatibility = next(iter(sols[0].compatibility))
    seen_einsums = {first_compatibility.einsum_id}
    seen_tensors = set(first_compatibility.tensors)
    seen_ranks = set(first_compatibility.ranks)

    def print(str):
        if verbose_stream is not None:
            verbose_stream.write(str + "\n")

    while compatibility_set_queue:
        einsum, next_sols = compatibility_set_queue.popleft()

        first_compatibility = next(iter(next_sols[0].compatibility))
        next_einsum = {first_compatibility.einsum_id}
        next_tensors = first_compatibility.tensors
        next_ranks = first_compatibility.ranks

        print("\n\n")
        print("\n\n" + "=" * 100 + f"\nProcessing einsum {einsum}")

        unseen_einsums = set(s for s, _ in compatibility_set_queue) | next_einsum
        unseen_tensors = (
            set.union(set(), *[s[1][0].tensors for s in compatibility_set_queue])
            | next_tensors
        )
        unseen_ranks = (
            set.union(set(), *[s[1][0].ranks for s in compatibility_set_queue])
            | next_ranks
        )

        def get_sols_a():
            prev_buckets = Cascade.bucket(
                sols, unseen_einsums, unseen_tensors, unseen_ranks
            )

            next_buckets = Cascade.bucket(
                next_sols, seen_einsums, seen_tensors, seen_ranks
            )

            print(f"Number of buckets: {len(prev_buckets)} x {len(next_buckets)}")

            # We can vertical combine the previous buckets now because we know that the choice within
            # each bucket is independent of future choices.
            Cascade.call_on_buckets(
                prev_buckets, lambda x: [Cascade.vertical_combine(x)]
            )

            new_sols = []
            for s1, s2 in Cascade.pair_matching_buckets(prev_buckets, next_buckets):
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
            prev_buckets = Cascade.bucket_multi_level(
                sols, unseen_einsums, unseen_tensors, unseen_ranks
            )

            next_buckets = Cascade.bucket_multi_level(
                next_sols, seen_einsums, seen_tensors, seen_ranks
            )

            # print(f"Part 2")
            Cascade.call_on_buckets(prev_buckets, Cascade.vertical_combine)
            new_sols = []

            for s1, s2 in Cascade.pair_matching_buckets(prev_buckets, next_buckets):
                new_sols.append(s1.combine(s2))
            import joblib

            # new_sols = joblib.Parallel(n_jobs=64)(
            #     [
            #         joblib.delayed(lambda x1, x2: x1.combine(x2))(x1, x2)
            #         for x1, x2 in Cascade.pair_matching_buckets(
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

        ops_left = set(s for s, _ in compatibility_set_queue)
        relevant_tensors = set.union(
            set(), *[s[1][0].tensors for s in compatibility_set_queue]
        )
        relevant_ranks = set.union(
            set(), *[s[1][0].ranks for s in compatibility_set_queue]
        )

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
