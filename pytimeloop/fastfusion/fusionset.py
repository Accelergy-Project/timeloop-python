from typing import Any, Generator, Union

import pandas as pd
from pareto import Pareto
from compatibility import OpCompatibility, TensorTiling, Loop, SharedResource
from collections import defaultdict
import unittest
import itertools
from util import fzs


class FusionSet:
    def __init__(
        self,
        compatibility: set[OpCompatibility],
        payload: Union[dict[fzs[str], Pareto], Pareto],
        shared_resources: list[SharedResource],
    ):
        self.compatibility: set[OpCompatibility] = compatibility
        self.payload: dict[str, Pareto] = {}
        self.shared_resources: list[SharedResource] = shared_resources

        if isinstance(payload, dict):
            self.payload = payload
        else:
            assert len(compatibility) == 1, "Payload must be a dict if multiple einsums"
            self.payload[fzs((next(iter(compatibility)).einsum_id,))] = payload

    def combine(self, other: "FusionSet"):
        return FusionSet(
            self.compatibility | other.compatibility, {**self.payload, **other.payload}
        )

    def compatible_with(self, other: "FusionSet") -> bool:
        return all(
            c.compatible_with(c2)
            for c in self.compatibility
            for c2 in other.compatibility
        )

    def relevant_compatibility(self, live_tensors: set[str]) -> "FusionSet":
        # Important aspects:
        # - What einsums are live (connected to a live einsum OR tiled fused with a live
        #   einsum): Effects how things are combined (can't compute a max with any
        #   still-live einsums)
        # - Relevant tensors: May be used in future decisions
        # - Relevant ranks: May be used in future decisions
        compatibility = {
            c.drop_dead(live_tensors)
            for c in self.compatibility
            if c.tensors & live_tensors
        }
        return OpCompatibility.vertical_combine(compatibility)

    @staticmethod
    def combine_combineable(fusion_sets: list["FusionSet"]) -> "FusionSet":
        if len(fusion_sets) == 1:
            return [fusion_sets[0]]
        buckets = defaultdict(list)
        for fs in fusion_sets:
            buckets[
                (
                    fzs(fs.compatibility),
                    fzs(fs.payload.keys()),
                    fzs(fs.shared_resources),
                )
            ].append(fs)
        return [FusionSet.vertical_combine(v) for v in buckets.values()]

    def drop_dead(self, live_tensors: set[str]):
        # Drop all dead compatibility
        new_compatibility = {c for c in self.compatibility if c.tensors & live_tensors}
        live_ops = {c.einsum_id for c in new_compatibility}

        # Combine dead payloads
        new_payload_keys = []
        payload_keys = list(self.payload.keys())
        cd = {c.einsum_id: c for c in self.compatibility}
        while payload_keys:
            to_check = [payload_keys.pop()]
            new_key = set(to_check[0])
            while to_check:
                t = to_check.pop()
                for k2 in payload_keys:
                    for c0, c1 in itertools.product(t, k2):
                        if cd[c0].co_tiled_with(cd[c1]):
                            new_key |= k2
                            to_check.append(k2)
                            payload_keys.remove(k2)
                            break
            new_payload_keys.append(fzs(new_key))

        new_payload = {}
        for k in new_payload_keys:
            new = Pareto.combine_live_all(
                p for k, p in self.payload.items() if k & new_key or k == new_key
            )
            k &= live_ops
            new = new_payload[k].combine_dead(new) if k in new_payload else new
            new_payload[k] = new

        self.payload = new_payload
        self.compatibility = {c.drop_dead(live_tensors) for c in new_compatibility}

    @property
    def tensors(self) -> set[str]:
        return set.union(*(set(c.tensors) for c in self.compatibility))

    @staticmethod
    def vertical_combine(fusion_sets: list["FusionSet"]) -> "FusionSet":
        if len(fusion_sets) == 1:
            return fusion_sets[0]
        fs = fusion_sets[0]
        keys = set(fzs(f.payload.keys()) for f in fusion_sets)
        assert len(keys) == 1, "Keys must be the same"
        return FusionSet(
            fs.compatibility,
            {
                k: Pareto.vertical_combine([f.payload[k] for f in fusion_sets])
                for k in fs.payload
            },
        )

    @staticmethod
    def bucket(
        fusion_sets: list["FusionSet"] | dict[Any, "FusionSet"],
        live_tensors: set[str],
    ) -> dict[fzs[OpCompatibility], list["FusionSet"] | dict]:
        if isinstance(fusion_sets, dict):
            return {
                k: FusionSet.bucket(v, live_tensors) for k, v in fusion_sets.items()
            }
        bucketed = defaultdict(list)
        for fs in fusion_sets:
            bucketed[fs.relevant_compatibility(live_tensors)].append(fs)
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
    def pair_matching_buckets_query(
        buckets_a: dict[fzs[OpCompatibility], list["FusionSet"]],
        buckets_b: dict[fzs[OpCompatibility], list["FusionSet"]],
    ):
        for k, v in buckets_a.items():
            if k in buckets_b:
                for v1, v2 in itertools.product(v, buckets_b[k]):
                    yield v1, v2

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
        c = " ".join(f"[{c}]" for c in self.compatibility)
        payload_keys = list(",".join(sorted(k)) for k in self.payload.keys())
        return c + " " + " ".join(f"[{k}]" for k in payload_keys)

    def __eq__(self, value: object) -> bool:
        return fzs(self.compatibility) == fzs(value.compatibility)

    def __lt__(self, other: "FusionSet") -> bool:
        return tuple(sorted(self.compatibility)) < tuple(sorted(other.compatibility))

    def __repr__(self):
        return f"FusionSet({self.compatibility})"


class TestFusionSet(unittest.TestCase):
    def test_vertical_combine(self):
        fs = []
        for i in range(2):
            comp = OpCompatibility(
                einsum_id=f"einsum1",
                tiling={},
            )
            fs.append(FusionSet({comp}, Pareto.get_dummy()))
        new_fs = FusionSet.vertical_combine(fs)
        self.assertEqual(len(new_fs.compatibility), 1)

    def test_combine(self):
        comp1 = OpCompatibility(
            einsum_id=f"einsum1",
            tiling={"Q": TensorTiling("GLB", (Loop("A", 1, False),))},
        )
        comp2 = OpCompatibility(
            einsum_id=f"einsum2",
            tiling={"V": TensorTiling("GLB", (Loop("A", 1, False),))},
        )
        fs1 = FusionSet({comp1}, Pareto.get_dummy())
        fs2 = FusionSet({comp2}, Pareto.get_dummy())
        new_fs = fs1.combine(fs2)
        self.assertEqual(len(new_fs.compatibility), 2)
        self.assertIn(comp1, new_fs.compatibility)
        self.assertIn(comp2, new_fs.compatibility)
        self.assertEqual(new_fs.tensors, {"Q", "V"})

    def test_compatibile_with(self):
        def get_tiling(rank_size: int):
            return {"T": TensorTiling("GLB", (Loop("A", rank_size, False),))}

        comp1 = OpCompatibility(einsum_id="A", tiling=get_tiling(1))
        comp2 = OpCompatibility(einsum_id="B", tiling=get_tiling(1))

        comp4 = OpCompatibility(einsum_id="C", tiling=get_tiling(1))
        comp5 = OpCompatibility(einsum_id="C", tiling=get_tiling(2))

        fs1 = FusionSet({comp1}, Pareto.get_dummy()).combine(
            FusionSet({comp2}, Pareto.get_dummy())
        )
        fs2 = FusionSet({comp4}, Pareto.get_dummy())
        self.assertEqual(fs1.compatible_with(fs2), True)

        fs1 = FusionSet({comp1}, Pareto.get_dummy())
        fs2 = FusionSet({comp5}, Pareto.get_dummy())
        # Not neighbors --> compatible becuase there's nothing overlapping to check
        self.assertEqual(fs1.compatible_with(fs2), False)

    # Test:
    # - Drop dead
    # - Finding live neighbors
    # -
    def test_drop_dead(self):
        comp1 = OpCompatibility(
            einsum_id=f"einsum1",
            tiling={"Q": TensorTiling("GLB", (Loop("A", 1, False),))},
        )
        comp2 = OpCompatibility(
            einsum_id=f"einsum2",
            tiling={"V": TensorTiling("GLB", (Loop("A", 1, False),))},
        )
        fs = FusionSet({comp1}, Pareto.get_dummy()).combine(
            FusionSet({comp2}, Pareto.get_dummy())
        )
        self.assertEqual(len(fs.compatibility), 2)
        fs.drop_dead({"Q"})
        self.assertEqual(len(fs.compatibility), 1)
        self.assertIn(comp1, fs.compatibility)
        fs.drop_dead(set())
        self.assertEqual(len(fs.compatibility), 0)

    def test_live_partition(self):
        ab = {"AB": TensorTiling("GLB", (Loop("AB", 1, False),))}
        bc = {"BC": TensorTiling("GLB", ())}
        cd = {"CD": TensorTiling("GLB", ())}
        de = {"DE": TensorTiling("GLB", (Loop("DE", 1, False),))}
        ef = {"EF": TensorTiling("GLB", (Loop("EF", 1, False),))}

        a = OpCompatibility(einsum_id="A", tiling={**ab})
        b = OpCompatibility(einsum_id="B", tiling={**ab, **bc})
        c = OpCompatibility(einsum_id="C", tiling={**bc, **cd})
        d = OpCompatibility(einsum_id="D", tiling={**cd, **de})
        e = OpCompatibility(einsum_id="E", tiling={**de, **ef})
        f = OpCompatibility(einsum_id="F", tiling={**ef})

        for live, partition in [
            (("AB",), ("", "AB")),
            (("BC",), ("", "B", "C")),
            (("CD",), ("", "C", "D")),
            (("DE",), ("", "DE")),
            (("EF",), ("", "EF")),
            (("AB", "EF"), ("", "AB", "EF")),
            (("BC", "EF"), ("B", "C", "EF")),
            ((), ("",)),
        ]:
            fs = FusionSet({a}, Pareto.get_dummy())
            for z in (b, c, d, e, f):
                fs = fs.combine(FusionSet({z}, Pareto.get_dummy()))
            fs.drop_dead(set(live))
            ids = tuple(sorted("".join(sorted(p for p in p2)) for p2 in fs.payload))
            partition = list(partition)
            partition = tuple(sorted(partition))
            msg = f"Failed with {live} {partition}, got {ids}"
            self.assertEqual(ids, partition, msg)


if __name__ == "__main__":
    unittest.main()
