from dataclasses import dataclass
from collections import defaultdict, namedtuple


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

    def matches(self, other: "OpCompatibility") -> bool:
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
        mine, other = self.fused_loops, other.fused_loops
        if mine and other:
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

    def combine(self, other: "OpCompatibility") -> "OpCompatibility":
        raise NotImplementedError()

    def drop_irrelevant(
        self, relevant_tensors: set[str], relevant_ranks: set[str]
    ) -> "OpCompatibility":
        return OpCompatibility(
            fused_tensors=self.fused_tensors & relevant_tensors,
            fused_loops=tuple(
                (l, f) for l, f in self.fused_loops if l in relevant_ranks
            ),
            fused_ranks=frozenset(self.fused_ranks & relevant_ranks),
            ranks=self.ranks & relevant_ranks,
            tensors=self.tensors & relevant_tensors,
            neighbors=self.neighbors,
        )

    def __eq__(self, other):
        return (
            self.fused_tensors == other.fused_tensors
            and len(self.fused_loops) == len(other.fused_loops)
            and all(
                len(a) == len(b) for a, b in zip(self.fused_loops, other.fused_loops)
            )
        )

    def __str__(self):
        return f"FTN: {self.fused_tensors}, FL: {self.fused_loops}"

    def __repr__(self):
        return str(self)

    # def __repr__(self):
    #     return f"OpCompatibility(fused_tensors={self.fused_tensors}, fused_loops={self.fused_loops}, ranks={self.ranks}, tensors={self.tensors})"

    def live_with(
        self, live_ops: set[str], live_compatibilities: dict[str, "OpCompatibility"]
    ) -> bool:
        return self.neighbors & live_ops or any(
            self.fused_ranks & c.fused_ranks for c in live_compatibilities.values()
        )


class MultiOpCompatibility:
    def __init__(self, compatibilities: dict[str, OpCompatibility]):
        self.compatibilities = compatibilities

    def matches(self, other: "MultiOpCompatibility", overlap: dict[str, str]) -> bool:
        for k, v in overlap.items():
            if not self.compatibilities[k].matches(other.compatibilities[v]):
                return False
        return True

    def combine(self, other: "MultiOpCompatibility") -> "MultiOpCompatibility":
        return MultiOpCompatibility({**self.compatibilities, **other.compatibilities})


class Payload:
    def __init__(self):
        pass

    def combine(self, other: "Payload"):
        return Payload()


class FusionSet:
    def __init__(self, compatibility: dict[str:OpCompatibility], payload: Payload):
        self.compatibility: dict[str, OpCompatibility] = compatibility
        self.payload: Payload = payload

    def combine(self, other: "FusionSet"):
        return FusionSet(
            compatibility={**self.compatibility, **other.compatibility},
            payload=self.payload.combine(
                other.payload
            ),  # TODO: INCLUDE SOMETHING TO DELAY PARETO-FINDING UNTIL ALL TILED FUSION IS RESOLVED
        )

    def matches(self, other: "FusionSet") -> bool:
        for c in self.compatibility.values():
            for n in c.neighbors:
                if n in other.compatibility and not c.matches(other.compatibility[n]):
                    return False
        return True

    def drop_irrelevant(self, relevant_tensors: set[str], relevant_ranks: set[str]):
        self.compatibility = {
            k: v.drop_irrelevant(relevant_tensors, relevant_ranks)
            for k, v in self.compatibility.items()
        }

    def as_tuple(self):
        return tuple(
            (k, self.compatibility[k]) for k in sorted(self.compatibility.keys())
        )

    def relevant_tuple(
        self,
        relevant_ops: set[str],
        relevant_tensors: set[str],
        relevant_ranks: set[str],
    ):
        live, newlive = None, {}
        while newlive != live:
            live = newlive
            newlive = {
                k: v
                for k, v in self.compatibility.items()
                if v.live_with(relevant_ops, live)
            }

        return tuple(
            (k, self.compatibility[k].drop_irrelevant(relevant_tensors, relevant_ranks))
            for k, v in live.items()
        )

    def vertical_combine(self, others: list["FusionSet"]):
        return self

    def __str__(self):
        return " ".join(f"[{k} {v}]" for k, v in self.compatibility.items())


if __name__ == "__main__":
    from more_itertools import powerset
    from itertools import permutations

    rank_sizes = {
        "matmul_1_M": 4,
        "matmul_1_K": 4,
        "matmul_1_N": 4,
        "matmul_2_N": 4,
        "matmul_3_N": 4,
    }
    fusable_tensors = {"A1", "C1", "C2", "C3"}
    must_fuse = {"C1"}

    compatibility_sets = []

    def get_compatibility_sets(
        op_name: str, neighbors: set[str], tensor2rank: dict[str, set[str]]
    ):
        print(f"Tensor2Rank: {tensor2rank}")

        compatibility_sets = []
        tensors = set(tensor2rank.keys())
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
                                fused_loops=tuple(
                                    (p, f) for p, f in zip(perm, factors)
                                ),
                                fused_tensors=frozenset(tn),
                                fused_ranks=frozenset(r),
                                ranks=frozenset(ranks),
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
        return op_name, [FusionSet({op_name: c}, Payload()) for c in compatibility_sets]

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
            {"M2"},
            {
                "C2": {"matmul_1_M", "matmul_2_N"},
                "B3": {"matmul_2_N", "matmul_3_N"},
                "C3": {"matmul_1_M", "matmul_3_N"},
            },
        ),
    ]

    op, sols = compatibility_sets.pop(0)
    ops = [op]
    for s in sols:
        print(s)

    while compatibility_sets:
        print("\n\n")

        # Put together the next set of solutions
        op, next_sols = compatibility_sets.pop(0)
        new_sols = []
        for s in sols:
            for ns in next_sols:
                if s.matches(ns):
                    new_sols.append(s.combine(ns))
                    # print(f"Combining {s} with {ns}")
                    # print(f"{s.combine(ns)}")
        sols = new_sols
        ops.append(op)

        # Prune irrelevant ranks and tensors
        if compatibility_sets:
            relevant_tensors = set.union(
                *[
                    set(c.tensors)
                    for _, s in compatibility_sets
                    for s2 in s
                    for c in s2.compatibility.values()
                ]
            )
            relevant_ranks = set.union(
                *[
                    set(c.ranks)
                    for _, s in compatibility_sets
                    for s2 in s
                    for c in s2.compatibility.values()
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
                s.relevant_tuple(ops_left, relevant_tensors, relevant_ranks)
            ].append(s)

        new_sols = []
        for k, v in bucketed.items():
            print(f"{len(v)} Bucket\n\t- {'\n\t- '.join(str(s) for s in k)}")
            for i in v:
                print(f"\t\t\t{i}")
            if len(v) > 1:
                new_sols.append(v[0].vertical_combine(v[1:]))
            else:
                new_sols.append(v[0])
        sols = new_sols

# TODO:
# - Feedback from compatibility sets -> payloads whether things are additive/maximal
# - If things are slowing down as we create larger fused sets, can collapse
#   no-longer-relevent ops into a dead ops group
