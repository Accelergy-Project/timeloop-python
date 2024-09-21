from dataclasses import dataclass
from collections import defaultdict
import itertools
from typing import Any, Generator


class Payload:
    def __init__(self):
        pass

    def combine(self, other: "Payload"):
        return Payload()


@dataclass(frozen=True)
class OpCompatibility:
    # Fusion information
    fused_tensors: frozenset[str]
    fused_loops: tuple[tuple[str, int]]
    fused_ranks: frozenset[str]

    # General information about the operation
    ranks: frozenset[str]
    tensors: frozenset[str]
    neighbors: frozenset[str]
    op_name: str

    def get_relevant_fused_loops(
        self, relevant_ranks: set[str]
    ) -> tuple[tuple[str, int]]:
        for i in range(len(self.fused_loops) - 1, -1, -1):
            if self.fused_loops[i][0] in relevant_ranks:
                return self.fused_loops[: i + 1]
        return ()

    def compatible_with(self, other: "OpCompatibility") -> bool:
        # Incompatible if one operation fuses a tensor that the other operation
        # has & does not fuse.
        for a, b in [(self, other), (other, self)]:
            if (a.fused_tensors - b.fused_tensors) & b.tensors:
                return False

        # Trivially compatible if there are no fused tensors between the two
        if not self.fused_tensors & other.fused_tensors:
            return True

        # Check tiled fused compatibility
        # Assume operation B fuses bigger or iso-size tiles as operation S.
        # - B & S must exchange tiles in same order -> Rank names must match in
        #   order.
        # - B tiles must be divisible by S tiles -> Rank shapes must match
        #   exactly, except for the innermost loop of B, where the loop bound of
        #   S must be divisible by the loop bound of B (meaning that the tile
        #   size of L is a multiple of the tile size of S).
        mine = self.get_relevant_fused_loops(other.ranks)
        other = other.get_relevant_fused_loops(self.ranks)
        if mine and other:
            # We're only worried about loops up to the innermost shared rank
            big_tiler, small_tiler = mine, other  # Default
            if len(mine) > len(other):  # Further subdivide -> smaller tile
                big_tiler, small_tiler = other, mine
            elif len(other) > len(mine):  # Further subdivide -> smaller tile
                big_tiler, small_tiler = mine, other
            elif mine[-1][-1] > other[-1][-1]:  # Larger innermost loop -> smaller tile
                big_tiler, small_tiler = other, mine

            for i, (s, l) in enumerate(zip(small_tiler, big_tiler)):
                if s[0] != l[0]:
                    return False
                if i < len(big_tiler) - 1 and s[1] != l[1]:
                    return False
                if i == len(big_tiler) - 1 and s[1] % l[1] != 0:
                    return False

        return True

    def drop_irrelevant(
        self,
        relevant_tensors: set[str] = frozenset(),
        relevant_ranks: set[str] = frozenset(),
        ignore_rank_sizes: bool = False,
    ) -> "OpCompatibility":
        # Relevant fused loops are all loops that are fused with a
        # relevant rank OR are above a loop with a relevant rank.
        fused_loops = self.get_relevant_fused_loops(relevant_ranks)
        if ignore_rank_sizes:
            fused_loops = tuple((r, 1) for r, _ in fused_loops)

        return OpCompatibility(
            op_name=self.op_name,
            fused_tensors=self.fused_tensors & relevant_tensors,
            fused_loops=fused_loops,
            fused_ranks=frozenset(r for r, _ in fused_loops),
            ranks=self.ranks & relevant_ranks,
            tensors=self.tensors & relevant_tensors,
            neighbors=self.neighbors,
        )

    @staticmethod
    def get_live(
        compats: set["OpCompatibility"],
        ops: set[str] = frozenset(),
    ) -> set["OpCompatibility"]:
        # Live are:
        # - All neighbors of an original node
        # - All tiled fused with a live node
        live = set()
        to_check = [c for c in compats if c.op_name in ops or c.neighbors & ops]
        while to_check:
            c = to_check.pop()
            live.add(c)
            to_check.extend(
                c2
                for c2 in compats
                if c2.op_name in c.neighbors
                and c2.fused_ranks & c.fused_ranks
                and c2 not in live
                and c2 not in to_check
            )
        return live

    @staticmethod
    def get_live_partitions(
        compats: set["OpCompatibility"],
    ) -> list[set["OpCompatibility"]]:
        partitions = []
        while compats:
            c = next(iter(compats))
            live = OpCompatibility.get_live(compats, {c.op_name})
            assert c in live
            partitions.append(live)
            compats -= live
        return partitions

    def __eq__(self, other):
        return (
            self.op_name == other.op_name
            and self.fused_tensors == other.fused_tensors
            and len(self.fused_loops) == len(other.fused_loops)
            and all(
                len(a) == len(b) for a, b in zip(self.fused_loops, other.fused_loops)
            )
        )

    def __lt__(self, other):
        return (
            self.op_name < other.op_name
            or self.fused_tensors < other.fused_tensors
            or self.fused_loops < other.fused_loops
        )

    def __str__(self):
        return (
            f"{self.op_name} ftensors {self.fused_tensors}, floops {self.fused_loops}"
        )

    def __repr__(self):
        return (
            f"OpCompatibility({self.op_name}, {self.fused_tensors}, {self.fused_loops})"
        )


class FusionSet:
    def __init__(self, compatibility: set[OpCompatibility], payload: Payload):
        self.compatibility: set[OpCompatibility] = compatibility
        self.payload: Payload = payload
        self.compatibility_dict = {c.op_name: c for c in compatibility}

    def combine(self, other: "FusionSet"):
        payload = other.payload if self.payload is None else self.payload
        if self.payload and other.payload:
            payload = self.payload.combine(other.payload)
        compatibility = self.compatibility | other.compatibility
        return FusionSet(compatibility, payload)

    def compatible_with(self, other: "FusionSet") -> bool:
        for c in self.compatibility_dict.values():
            for n in c.neighbors:
                if n in other.compatibility_dict and not c.compatible_with(
                    other.compatibility_dict[n]
                ):
                    return False
        return True

    def drop_irrelevant(self, relevant_tensors: set[str], relevant_ranks: set[str]):
        self.compatibility = {
            c.drop_irrelevant(relevant_tensors, relevant_ranks)
            for c in self.compatibility
        }
        self.compatibility_dict = {c.op_name: c for c in self.compatibility}

    def relevant_compatibility(
        self,
        live_ops: set[str],
        relevant_tensors: set[str],
        relevant_ranks: set[str],
        ignore_rank_sizes=False,
        ignore_live_not_neighbors=False,
    ) -> "FusionSet":
        # Important aspects:
        # - What ops are live (connected to a live op OR tiled fused with a live
        #   op): Effects how things are combined (can't compute a max with any
        #   still-live ops)
        # - Relevant tensors: May be used in future decisions
        # - Relevant ranks: May be used in future decisions
        my_live = OpCompatibility.get_live(self.compatibility, live_ops)

        immediate_neighbors = {c for c in my_live if live_ops & c.neighbors}
        live_not_neighbors = my_live - immediate_neighbors
        if ignore_live_not_neighbors:
            live_not_neighbors = set()

        neighbors = {
            c.drop_irrelevant(relevant_tensors, relevant_ranks, ignore_rank_sizes)
            for c in immediate_neighbors
        }
        live = {c.drop_irrelevant() for c in live_not_neighbors}
        return FusionSet(neighbors | live, None)

    def drop_dead(self, live_ops: set[str]):
        live = OpCompatibility.get_live(self.compatibility, live_ops)
        self.compatibility = live
        self.compatibility_dict = {c.op_name: c for c in live}

    @property
    def tensors(self) -> set[str]:
        return set.union(*(set(c.tensors) for c in self.compatibility))

    @property
    def ranks(self) -> set[str]:
        return set.union(*(set(c.ranks) for c in self.compatibility))

    @staticmethod
    def vertical_combine(fusion_sets: list["FusionSet"]):
        # TODO
        return fusion_sets[0]

    @staticmethod
    def bucket(
        fusion_sets: list["FusionSet"] | dict[Any, "FusionSet"],
        live_ops: set[str],
        relevant_tensors: set[str],
        relevant_ranks: set[str],
        ignore_rank_sizes=False,
        ignore_live_not_neighbors=False,
    ) -> dict[frozenset[OpCompatibility], list["FusionSet"] | dict]:
        args = (
            live_ops,
            relevant_tensors,
            relevant_ranks,
            ignore_rank_sizes,
            ignore_live_not_neighbors,
        )

        if isinstance(fusion_sets, dict):
            return {k: FusionSet.bucket(v, *args) for k, v in fusion_sets.items()}

        bucketed = defaultdict(list)
        for fs in fusion_sets:
            bucketed[fs.relevant_compatibility(*args)].append(fs)
        return bucketed

    @staticmethod
    def bucket_multi_level(
        fusion_sets: list["FusionSet"],
        live_ops: set[str],
        relevant_tensors: set[str],
        relevant_ranks: set[str],
    ) -> dict[frozenset[OpCompatibility], list["FusionSet"] | dict]:
        # First bucketing: Tensors only
        kwargs = dict(
            live_ops=live_ops,
            relevant_tensors=set(),
            relevant_ranks=set(),
            ignore_rank_sizes=True,
            ignore_live_not_neighbors=True,
        )
        bucketed = fusion_sets
        kwargs["relevant_tensors"] = relevant_tensors
        bucketed = FusionSet.bucket(bucketed, **kwargs)
        kwargs["relevant_ranks"] = relevant_ranks
        bucketed = FusionSet.bucket(bucketed, **kwargs)
        kwargs["ignore_rank_sizes"] = False
        bucketed = FusionSet.bucket(bucketed, **kwargs)
        kwargs["ignore_live_not_neighbors"] = False
        bucketed = FusionSet.bucket(bucketed, **kwargs)

        return bucketed

    @staticmethod
    def pair_matching_buckets(
        buckets_a: dict[frozenset[OpCompatibility], list["FusionSet"] | dict],
        buckets_b: dict[frozenset[OpCompatibility], list["FusionSet"] | dict],
    ) -> Generator[tuple["FusionSet", "FusionSet"], None, None]:
        def cast(x):
            if isinstance(x, dict):
                return x.items()
            if isinstance(x, list):
                return [(v, v) for v in x]
            if isinstance(x, FusionSet):
                return [(x, x)]
            raise ValueError(f"Invalid type {type(x)}")

        def _pair_recursive(a, b):
            if isinstance(a, FusionSet) and isinstance(b, FusionSet):
                yield a, b
                return
            for (k1, v1), (k2, v2) in itertools.product(cast(a), cast(b)):
                if k1.compatible_with(k2):
                    yield from _pair_recursive(v1, v2)

        yield from _pair_recursive(buckets_a, buckets_b)

    @staticmethod
    def call_on_buckets(
        buckets: dict[frozenset[OpCompatibility], list["FusionSet"] | dict],
        func,
    ):
        def _call_recursive(b):
            if isinstance(b, dict):
                for k, v in b.items():
                    b[k] = _call_recursive(v)
                return b
            return func(b)

        _call_recursive(buckets)

    def __hash__(self) -> int:
        return hash(frozenset(self.compatibility))

    def __str__(self):
        return " ".join(f"[{c}]" for c in self.compatibility)

    def __eq__(self, value: object) -> bool:
        return frozenset(self.compatibility) == frozenset(value.compatibility)

    def __lt__(self, other: "FusionSet") -> bool:
        return self.compatibility < other.compatibility

    def __repr__(self):
        return f"FusionSet({self.compatibility})"


if __name__ == "__main__":
    from more_itertools import powerset
    from itertools import permutations

    rank_sizes = {
        "matmul_1_M": 1024,
        "matmul_1_K": 1024,
        "matmul_1_N": 1024,
        "matmul_2_N": 1024,
        "matmul_3_N": 1024,
        "matmul_4_N": 1024,
    }
    fusable_tensors = {"A1", "C1", "C2", "C3", "C4"}
    must_fuse = set()  # set(["C1", "C2", "C3", "C4"])

    compatibility_sets = []

    def get_compatibility_sets(
        op_name: str, neighbors: set[str], tensor2rank: dict[str, set[str]]
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
                                op_name=op_name,
                                fused_loops=tuple(
                                    (p, f) for p, f in zip(perm, factors)
                                ),
                                fused_tensors=frozenset(tn),
                                fused_ranks=frozenset(r),
                                ranks=frozenset(all_ranks),
                                tensors=frozenset(tensors),
                                neighbors=frozenset(neighbors),
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
        return op_name, [FusionSet({c}, Payload()) for c in compatibility_sets]

    compatibility_sets = [
        get_compatibility_sets(
            "M1",
            {"M2"},
            {
                "A1": {"matmul_1_M", "matmul_1_K"},
                "B1": {"matmul_1_K", "matmul_1_N"},
                "C1": {"matmul_1_M", "matmul_1_N"},
            },
        ),
        get_compatibility_sets(
            "M2",
            {"M1", "M3"},
            {
                "C1": {"matmul_1_M", "matmul_1_N"},
                "B2": {"matmul_1_N", "matmul_2_N"},
                "C2": {"matmul_1_M", "matmul_2_N"},
            },
        ),
        get_compatibility_sets(
            "M3",
            {"M2", "M4"},
            {
                "C2": {"matmul_1_M", "matmul_2_N"},
                "B3": {"matmul_2_N", "matmul_3_N"},
                "C3": {"matmul_1_M", "matmul_3_N"},
            },
        ),
        get_compatibility_sets(
            "M4",
            {"M3"},
            {
                "C3": {"matmul_1_M", "matmul_3_N"},
                "B4": {"matmul_3_N", "matmul_4_N"},
                "C4": {"matmul_1_M", "matmul_4_N"},
            },
        ),
    ]

    op, sols = compatibility_sets.pop(0)
    ops = [op]
    first_compatibility = next(iter(sols[0].compatibility))
    seen_ops = {first_compatibility.op_name}
    seen_tensors = set(first_compatibility.tensors)
    seen_ranks = set(first_compatibility.ranks)

    while compatibility_sets:
        # Put together the next set of solutions
        op, next_sols = compatibility_sets.pop(0)

        first_compatibility = next(iter(next_sols[0].compatibility))
        next_op = {first_compatibility.op_name}
        next_tensors = first_compatibility.tensors
        next_ranks = first_compatibility.ranks

        print("\n\n")
        print("\n\n" + "=" * 100 + f"\nProcessing op {op}")

        if compatibility_sets:
            unseen_ops = set(s for s, _ in compatibility_sets) | next_op
            unseen_tensors = (
                set.union(*[s[1][0].tensors for s in compatibility_sets]) | next_tensors
            )
            unseen_ranks = (
                set.union(*[s[1][0].ranks for s in compatibility_sets]) | next_ranks
            )
        else:
            unseen_ops = set() | next_op
            unseen_tensors = set() | next_tensors
            unseen_ranks = set() | next_ranks

        # Further speed increase:
        # - We can keep this bucketing for later when we do vertical combining. Need to do a second
        #   bucketing by live sets
        def get_sols_a():
            prev_buckets = FusionSet.bucket(
                sols, unseen_ops, unseen_tensors, unseen_ranks
            )

            next_buckets = FusionSet.bucket(
                next_sols, seen_ops, seen_tensors, seen_ranks
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
                sols, unseen_ops, unseen_tensors, unseen_ranks
            )

            next_buckets = FusionSet.bucket_multi_level(
                next_sols, seen_ops, seen_tensors, seen_ranks
            )

            # print(f"Part 2")
            FusionSet.call_on_buckets(prev_buckets, FusionSet.vertical_combine)
            new_sols = []
            for s1, s2 in FusionSet.pair_matching_buckets(prev_buckets, next_buckets):
                new_sols.append(s1.combine(s2))

            print(f"C: Generated {len(new_sols)} from {len(sols)} x {len(next_sols)}")
            return new_sols

        [s.drop_dead(unseen_ops) for s in sols]

        import timeit

        print(f"Time A: {timeit.timeit(get_sols_a, number=1)}")
        # print(f"Time B: {timeit.timeit(get_sols_b, number=1)}")
        print(f"Time C: {timeit.timeit(get_sols_c, number=1)}")

        sols = get_sols_c()
        print(f"Generated {len(sols)} solutions")
        ops.append(op)
        # assert False

        # Prune irrelevant ranks and tensors
        if compatibility_sets:
            relevant_tensors = set.union(
                *[
                    set(c.tensors)
                    for _, s in compatibility_sets
                    for s2 in s
                    for c in s2.compatibility
                ]
            )
            relevant_ranks = set.union(
                *[
                    set(c.ranks)
                    for _, s in compatibility_sets
                    for s2 in s
                    for c in s2.compatibility
                ]
            )
            ops_left = set(s for s, _ in compatibility_sets)
        else:
            relevant_tensors = set()
            relevant_ranks = set()
            ops_left = set()

        print("\n\n")
        print(f"Relevant Tensors: {relevant_tensors}")
        print(f"Relevant Ranks: {relevant_ranks}")
        # for s in sols:
        #     s.drop_irrelevant(relevant_tensors, relevant_ranks)

        bucketed = defaultdict(list)
        for s in sols:
            # ALSO NEED TO INCLDUE TILED FUSED
            bucketed[
                s.relevant_compatibility(ops_left, relevant_tensors, relevant_ranks)
            ].append(s)

        # new_sols = []
        # for k, v in sorted(bucketed.items()):
        #     print(f"{len(v)} Bucket\n\t- {'\n\t- '.join(str(s) for s in k)}")
        #     for i in sorted(v):
        #         print(f"\t\t\t{sorted(i.compatibility)}")
        #     if len(v) > 1:
        #         new_sols.append(v[0].vertical_combine(v[1:]))
        #     else:
        #         new_sols.append(v[0])
        # sols = new_sols
        seen_ops |= next_op
        seen_tensors |= next_tensors
        seen_ranks |= next_ranks

# TODO:
# - Feedback from compatibility sets -> payloads whether things are additive/maximal
