from typing import Any, Generator
from pareto import Pareto
from compatibility import OpCompatibility
from collections import defaultdict
import unittest
import itertools
from util import fzs


class FusionSet:
    def __init__(self, compatibility: set[OpCompatibility], payload: Pareto):
        self.compatibility: set[OpCompatibility] = compatibility
        self.payload: Pareto = payload
        self.compatibility_dict = {c.einsum_id: c for c in compatibility}

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
        self.compatibility_dict = {c.einsum_id: c for c in self.compatibility}

    def relevant_compatibility(
        self,
        live_einsums: set[str],
        relevant_tensors: set[str],
        relevant_ranks: set[str],
        ignore_rank_sizes=False,
        ignore_live_not_neighbors=False,
    ) -> "FusionSet":
        # Important aspects:
        # - What einsums are live (connected to a live einsum OR tiled fused with a live
        #   einsum): Effects how things are combined (can't compute a max with any
        #   still-live einsums)
        # - Relevant tensors: May be used in future decisions
        # - Relevant ranks: May be used in future decisions
        my_live = OpCompatibility.get_live(self.compatibility, live_einsums)

        immediate_neighbors = {
            c
            for c in my_live
            if live_einsums & c.neighbors or c.einsum_id in live_einsums
        }
        live_not_neighbors = my_live - immediate_neighbors
        if ignore_live_not_neighbors:
            live_not_neighbors = set()

        neighbors = {
            c.drop_irrelevant(relevant_tensors, relevant_ranks, ignore_rank_sizes)
            for c in immediate_neighbors
        }
        live = {c.drop_irrelevant() for c in live_not_neighbors}
        return FusionSet(neighbors | live, None)

    def drop_dead(self, live_einsums: set[str]):
        tiled_partitions = OpCompatibility.get_tiled_partitions(self.compatibility)
        live_partitions, dead_partitions = [], []
        for t in tiled_partitions:
            if any(
                p.einsum_id in live_einsums or p.neighbors & live_einsums for p in t
            ):
                live_partitions.append(t)
            else:
                dead_partitions.append(t)

        self.payload = self.payload.drop_dead(live_partitions, dead_partitions)
        live_ids = [p.einsum_id for p2 in live_partitions for p in p2]
        self.compatibility_dict = {p: self.compatibility_dict[p] for p in live_ids}
        self.compatibility = set(self.compatibility_dict.values())

    @property
    def tensors(self) -> set[str]:
        return set.union(*(set(c.tensors) for c in self.compatibility))

    @property
    def ranks(self) -> set[str]:
        return set.union(*(set(c.ranks) for c in self.compatibility))

    @staticmethod
    def vertical_combine(fusion_sets: list["FusionSet"]) -> "FusionSet":
        return FusionSet(
            fusion_sets[0].compatibility,
            Pareto.vertical_combine([f.payload for f in fusion_sets]),
        )

    @staticmethod
    def bucket(
        fusion_sets: list["FusionSet"] | dict[Any, "FusionSet"],
        live_einsums: set[str],
        relevant_tensors: set[str],
        relevant_ranks: set[str],
        ignore_rank_sizes=False,
        ignore_live_not_neighbors=False,
    ) -> dict[fzs[OpCompatibility], list["FusionSet"] | dict]:
        args = (
            live_einsums,
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
        live_einsums: set[str],
        relevant_tensors: set[str],
        relevant_ranks: set[str],
    ) -> dict[fzs[OpCompatibility], list["FusionSet"] | dict]:
        # First bucketing: Tensors only
        kwargs = dict(
            live_einsums=live_einsums,
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
        buckets_a: dict[fzs[OpCompatibility], list["FusionSet"] | dict],
        buckets_b: dict[fzs[OpCompatibility], list["FusionSet"] | dict],
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
        buckets: dict[fzs[OpCompatibility], list["FusionSet"] | dict],
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
        return hash(fzs(self.compatibility))

    def __str__(self):
        return " ".join(f"[{c}]" for c in self.compatibility)

    def __eq__(self, value: object) -> bool:
        return fzs(self.compatibility) == fzs(value.compatibility)

    def __lt__(self, other: "FusionSet") -> bool:
        return self.compatibility < other.compatibility

    def __repr__(self):
        return f"FusionSet({self.compatibility})"


class TestFusionSet(unittest.TestCase):
    def test_vertical_combine(self):
        fs = []
        for i in range(2):
            comp = OpCompatibility(
                einsum_id=f"einsum1",
                fused_tensors=fzs(),
                fused_loops=(),
                fused_ranks=fzs(),
                ranks=fzs(),
                tensors=fzs(),
                neighbors=fzs(),
            )
            fs.append(FusionSet({comp}, Pareto(data={})))
        new_fs = FusionSet.vertical_combine(fs)
        self.assertEqual(len(new_fs.compatibility), 1)
        self.assertEqual(new_fs.payload.data, {})

    def test_combine(self):
        comp1 = OpCompatibility(
            einsum_id=f"einsum1",
            fused_tensors=fzs(),
            fused_loops=(),
            fused_ranks=fzs(),
            ranks=fzs("R"),
            tensors=fzs("Q"),
            neighbors=fzs("123"),
        )
        comp2 = OpCompatibility(
            einsum_id=f"einsum2",
            fused_tensors=fzs(),
            fused_loops=(),
            fused_ranks=fzs(),
            ranks=fzs("S"),
            tensors=fzs("V"),
            neighbors=fzs("ABC"),
        )
        fs1 = FusionSet({comp1}, Pareto(data={}))
        fs2 = FusionSet({comp2}, Pareto(data={}))
        new_fs = fs1.combine(fs2)
        self.assertEqual(len(new_fs.compatibility), 2)
        self.assertIn(comp1, new_fs.compatibility)
        self.assertIn(comp2, new_fs.compatibility)
        self.assertEqual(new_fs.payload.data, {})
        self.assertEqual(new_fs.tensors, {"Q", "V"})
        self.assertEqual(new_fs.ranks, {"R", "S"})

    def test_compatibile_with(self):
        for neighbors in fzs("ABC"), fzs():
            kwargs = dict(
                fused_tensors=fzs("T1"),
                fused_ranks=fzs(),
                ranks=fzs("A"),
                tensors=fzs(),
                neighbors=neighbors,
            )

            comp1 = OpCompatibility(einsum_id="A", fused_loops=(("A", 1),), **kwargs)
            comp2 = OpCompatibility(einsum_id="B", fused_loops=(("A", 2),), **kwargs)

            comp4 = OpCompatibility(einsum_id="C", fused_loops=(("A", 4),), **kwargs)
            comp5 = OpCompatibility(einsum_id="C", fused_loops=(("A", 3),), **kwargs)

            fs1 = FusionSet({comp1, comp2}, Pareto(data={}))
            fs2 = FusionSet({comp4}, Pareto(data={}))
            self.assertEqual(fs1.compatible_with(fs2), True)

            fs2 = FusionSet({comp5}, Pareto(data={}))
            # Not neighbors --> compatible becuase there's nothing overlapping to check
            self.assertEqual(fs1.compatible_with(fs2), not neighbors)

    # Test:
    # - Drop dead
    # - Finding live neighbors
    # -
    def test_drop_dead(self):
        comp1 = OpCompatibility(
            einsum_id=f"einsum1",
            fused_tensors=fzs(),
            fused_loops=(),
            fused_ranks=fzs(),
            ranks=fzs("R"),
            tensors=fzs("Q"),
            neighbors=fzs("123"),
        )
        comp2 = OpCompatibility(
            einsum_id=f"einsum2",
            fused_tensors=fzs(),
            fused_loops=(),
            fused_ranks=fzs(),
            ranks=fzs("S"),
            tensors=fzs("V"),
            neighbors=fzs("ABC"),
        )
        fs = FusionSet({comp1, comp2}, Pareto(data={}))
        fs.drop_dead({"einsum1"})
        self.assertEqual(len(fs.compatibility), 1)
        self.assertIn(comp1, fs.compatibility)
        self.assertEqual(fs.payload.data, {})
        fs.drop_dead(set())
        self.assertEqual(len(fs.compatibility), 0)
        self.assertEqual(fs.payload.data, {})

    def test_live_partition(self):
        kwargs = dict(
            fused_tensors=fzs("T1"),
            ranks=fzs("A"),
            tensors=fzs(),
            fused_loops=(),
        )

        a = OpCompatibility(
            einsum_id="A", fused_ranks=fzs("A"), neighbors=fzs("B"), **kwargs
        )
        b = OpCompatibility(
            einsum_id="B", fused_ranks=fzs("A"), neighbors=fzs("AC"), **kwargs
        )
        c = OpCompatibility(
            einsum_id="C", fused_ranks=fzs(), neighbors=fzs("BD"), **kwargs
        )
        d = OpCompatibility(
            einsum_id="D", fused_ranks=fzs("A"), neighbors=fzs("CE"), **kwargs
        )
        e = OpCompatibility(
            einsum_id="E", fused_ranks=fzs("A"), neighbors=fzs("DF"), **kwargs
        )
        f = OpCompatibility(
            einsum_id="F", fused_ranks=fzs("A"), neighbors=fzs("E"), **kwargs
        )

        for live, partition in [
            ("A", ("AB",)),
            ("B", ("AB", "C")),
            ("C", ("AB", "C", "DEF")),
            ("D", ("C", "DEF")),
            ("E", ("DEF",)),
            ("F", ("DEF",)),
            ("AF", ("AB", "DEF")),
            ("ABF", ("AB", "C", "DEF")),
        ]:
            fs = FusionSet({a, b, c, d, e, f}, Pareto(data={}))
            fs.drop_dead(set(live))
            partitions = OpCompatibility.get_tiled_partitions(fs.compatibility)
            ids = tuple(
                sorted("".join(sorted(p.einsum_id for p in p2)) for p2 in partitions)
            )
            msg = f"Failed with {live} {partition}, got {ids}"
            self.assertEqual(len(fs.compatibility), sum(len(l) for l in partition), msg)
            self.assertEqual(ids, partition, msg)

    def test_bucketing(self):
        tensor_choices = ["A", "B", "BC"]
        rank_choices = ["MN", "NM"]
        rank_size_choices = [1, 2]
        has_other_einsum = [True, False]

        comps = []
        for t in tensor_choices:
            for r in rank_choices:
                for rs in rank_size_choices:
                    for other in has_other_einsum:
                        kwargs = dict(
                            fused_tensors=fzs(t),
                            fused_loops=tuple((x, rs) for x in r),
                            fused_ranks=fzs(r),
                            ranks=fzs("MN"),
                            tensors=fzs(t),
                            neighbors=fzs(),
                        )
                        x = {OpCompatibility(einsum_id="einsum1", **kwargs)}
                        if other:
                            x.add(OpCompatibility(einsum_id="einsum2", **kwargs))
                        comps.append(FusionSet(x, Pareto(data={})))

        def check_bucket_sizes(bucketed, expected):
            if expected:
                self.assertEqual(len(bucketed), expected[0])
                for b in bucketed.values():
                    check_bucket_sizes(b, expected[1:])

        fusion_sets = FusionSet.bucket_multi_level(
            comps, {"einsum1"}, {"A"}, {"M", "N"}
        )
        check_bucket_sizes(fusion_sets, [2, 2, 2])
        fusion_sets = FusionSet.bucket_multi_level(
            comps, {"einsum1", "einsum2"}, {"A"}, {"M", "N"}
        )
        check_bucket_sizes(fusion_sets, [4, 2, 2])
        fusion_sets = FusionSet.bucket_multi_level(
            comps, {"einsum1"}, {"A", "C"}, {"M"}
        )
        check_bucket_sizes(fusion_sets, [3, 2, 2])
        fusion_sets = FusionSet.bucket_multi_level(
            comps, {"einsum1"}, {"A", "C"}, set()
        )
        check_bucket_sizes(fusion_sets, [3, 1, 1])
        fusion_sets = FusionSet.bucket_multi_level(
            comps, {"einsum1", "einsum2"}, set(), set()
        )
        check_bucket_sizes(fusion_sets, [2, 1, 1])


if __name__ == "__main__":
    unittest.main()
