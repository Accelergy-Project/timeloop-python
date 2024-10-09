from collections import deque
from itertools import permutations
import unittest

from more_itertools import powerset

from pytimeloop.fastfusion.compatibility import *
from pytimeloop.fastfusion.exploration import *
from pytimeloop.fastfusion.pareto import *


class TestExploration(unittest.TestCase):
    def test_exploration(self):
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

        explore_fusion_sets(deque(compatibility_sets))
