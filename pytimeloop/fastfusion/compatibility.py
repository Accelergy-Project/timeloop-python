from dataclasses import dataclass
from collections import namedtuple


@dataclass(frozen=True)
class OpCompatibility:
    # Fusion information
    fused_tensors: frozenset[str]
    fused_loops: tuple[tuple[str, int]]

    # General information about the operation
    ranks: frozenset[str]
    tensors: frozenset[str]

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
            fused_loops=[(l, f) for l, f in self.fused_loops if l in relevant_ranks],
            ranks=self.ranks & relevant_ranks,
            tensors=self.tensors & relevant_tensors,
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
        return f"Fused Tensors: {self.fused_tensors}, Fused Loops: {self.fused_loops}"

    def __repr__(self):
        return f"OpCompatibility(fused_tensors={self.fused_tensors}, fused_loops={self.fused_loops}, ranks={self.ranks}, tensors={self.tensors})"


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
        raise NotImplementedError()


class PotentialSolution:
    def __init__(self, compatibility: dict[str:OpCompatibility], payload: Payload):
        self.compatibility: dict[str, OpCompatibility] = compatibility
        self.payload: Payload = payload

    def combine(self, other: "PotentialSolution"):
        return PotentialSolution(
            compatibility={**self.compatibility, **other.compatibility},
            payload=self.payload.combine(other.payload),
        )

    def matches(
        self,
        other: "PotentialSolution",
        shared_ops: list[str],
        shared_ranks: dict[str, str],
    ) -> bool:
        return self.compatibility.matches(other.compatibility, shared_ranks)


class FusionSet:
    def __init__(self, solutions: dict[str, PotentialSolution]):
        self.solutions = solutions

    def combine(self, other: "Payload"):
        results = {}


if __name__ == "__main__":
    from more_itertools import powerset
    from itertools import permutations

    rank_sizes = {
        "matmul_1_M": 4,
        "matmul_1_K": 2,
        "matmul_1_N": 2,
        "matmul_2_N": 2,
        "matmul_3_N": 2,
    }
    fusable_tensors = {"A1", "C1", "C2", "C3"}
    must_fuse = {"A1", "C1", "C2", "C3"}

    def get_compatibility_sets(tensor2rank: dict[str, set[str]]):
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
                                fused_loops=[(p, f) for p, f in zip(perm, factors)],
                                fused_tensors=set(tn),
                                ranks=set(ranks),
                                tensors=tensors,
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
        return compatibility_sets

    sets1 = get_compatibility_sets(
        {
            "A1": {"matmul_1_M", "matmul_1_K"},
            "B1": {"matmul_1_K", "matmul_1_N"},
            "C1": {"matmul_1_M", "matmul_1_N"},
        }
    )
    sets2 = get_compatibility_sets(
        {
            "C1": {"matmul_1_M", "matmul_1_N"},
            "B2": {"matmul_1_N", "matmul_2_N"},
            "C2": {"matmul_1_M", "matmul_2_N"},
        }
    )
    sets3 = get_compatibility_sets(
        {
            "C2": {"matmul_1_M", "matmul_2_N"},
            "B3": {"matmul_2_N", "matmul_3_N"},
            "C3": {"matmul_1_M", "matmul_3_N"},
        }
    )
    for s1 in sets1:
        print(f"\n{s1}")
        for s2 in sets2:
            if not s1.matches(s2):
                continue
            print(f"\t{s2}")
            for s3 in sets3:
                if not s2.matches(s3):
                    continue
                # print(f"\t\t{s3}")

    ops = ["M1", "M2", "M3"]
    potential_solution = namedtuple("PotentialSolution", ops)
