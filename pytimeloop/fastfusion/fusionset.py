import functools
from typing import Any, Generator, Union

import pandas as pd
import tqdm
from pareto import Pareto
from compatibility import Compatibility, TensorTiling, Loop, SharedResource
from collections import defaultdict
import unittest
import itertools
from util import fzs


class InterchangeableSet:
    def __init__(
        self,
        compatibility: set[Compatibility],
        payload: Union[dict[str, Pareto], Pareto],
        shared_resources: Union[
            dict[str, list[SharedResource]], list[SharedResource]
        ] = (),
    ):
        self.compatibility: set[Compatibility] = compatibility

        def cast(x, name):
            if isinstance(x, dict):
                return x
            assert len(compatibility) == 1, f"{name} must be a dict if multiple einsums"
            return {next(iter(compatibility)).einsum_id: x}

        self.payload: dict[str, Pareto] = cast(payload, "payload")
        self.shared_resources: dict[str, list[SharedResource]] = cast(
            shared_resources, "shared_resources"
        )

    def compatible_with(self, other: "InterchangeableSet") -> bool:
        return all(
            c.compatible_with(c2)
            for c in self.compatibility
            for c2 in other.compatibility
        )

    def relevant_compatibility(self, live_tensors: set[str]) -> "InterchangeableSet":
        compatibility = {
            c.drop_dead(live_tensors)
            for c in self.compatibility
            if c.tensors & live_tensors
        }
        return Compatibility.vertical_combine(compatibility)

    def combine(self, other: "InterchangeableSet"):
        return InterchangeableSet(
            self.compatibility | other.compatibility,
            {**self.payload, **other.payload},
            {**self.shared_resources, **other.shared_resources},
        )

    @staticmethod
    def combine_combineable(
        fusion_sets: list["InterchangeableSet"] | dict[Any, list["InterchangeableSet"]]
    ) -> list["InterchangeableSet"]:
        if isinstance(fusion_sets, dict):
            return {
                k: InterchangeableSet.combine_combineable(v)
                for k, v in tqdm.tqdm(list(fusion_sets.items()), desc="Combining")
            }
        if len(fusion_sets) == 1:
            return [fusion_sets[0]]
        buckets = defaultdict(list)
        for fs in tqdm.tqdm(fusion_sets, desc="Combining", leave=False):
            buckets[fs._hash_keys()].append(fs)
        return [InterchangeableSet.vertical_combine(v) for v in buckets.values()]

    def drop_dead(self, live_tensors: set[str]):
        # Live: Connected to a live einsum OR tiled fused with a live einsum
        live, dead = set(), self.compatibility
        next_live = {c for c in dead if c.tensors & live_tensors}
        while next_live:
            live |= next_live
            next_live = {c for c in dead if c.co_tiled_with(next_live)}
            dead -= next_live

        # Apply shared resources
        for d in dead:
            if d.einsum_id not in self.shared_resources:
                continue
            resource = self.shared_resources.pop(d.einsum_id)
            for r, d2 in itertools.product(resource, dead):
                if r.n_loops_above < d2.n_shared_loops(d):
                    p = self.payload[d2.einsum_id]
                    self.payload[d2.einsum_id] = p.add_shared_resource(resource)

        # Combine dead payloads
        dead_payload_keys = set(self.payload.keys()) - set(c.einsum_id for c in live)
        dead_payloads = {self.payload.pop(k) for k in dead_payload_keys}
        if dead_payloads:
            self.payload[""] = Pareto.combine_all(dead_payloads)
        self.compatibility = live

    @property
    def tensors(self) -> set[str]:
        return set.union(*(set(c.tensors) for c in self.compatibility))

    @staticmethod
    def vertical_combine(
        fusion_sets: list["InterchangeableSet"],
    ) -> "InterchangeableSet":
        if len(fusion_sets) == 1:
            return fusion_sets[0]
        fs = fusion_sets[0]
        assert len(set(hash(f) for f in fusion_sets)) == 1, "Hashes must be match"
        return InterchangeableSet(
            fs.compatibility,
            {
                k: Pareto.vertical_combine([f.payload[k] for f in fusion_sets])
                for k in fs.payload
            },
        )

    @staticmethod
    def bucket(
        fusion_sets: list["InterchangeableSet"] | dict[Any, "InterchangeableSet"],
        live_tensors: set[str],
    ) -> dict[fzs[Compatibility], list["InterchangeableSet"] | dict]:
        if isinstance(fusion_sets, dict):
            return {
                k: InterchangeableSet.bucket(v, live_tensors)
                for k, v in fusion_sets.items()
            }
        bucketed = defaultdict(list)
        for fs in tqdm.tqdm(fusion_sets, desc="Bucketing"):
            bucketed[fs.relevant_compatibility(live_tensors)].append(fs)
        return bucketed

    @staticmethod
    def pair_matching_buckets(
        buckets_a: dict[fzs[Compatibility], list["InterchangeableSet"] | dict],
        buckets_b: dict[fzs[Compatibility], list["InterchangeableSet"] | dict],
    ) -> Generator[tuple["InterchangeableSet", "InterchangeableSet"], None, None]:
        def cast(x):
            if isinstance(x, dict):
                return x.items()
            if isinstance(x, list):
                return [(v, v) for v in x]
            if isinstance(x, InterchangeableSet):
                return [(x, x)]
            raise ValueError(f"Invalid type {type(x)}")

        def _pair_recursive(a, b):
            if isinstance(a, InterchangeableSet) and isinstance(b, InterchangeableSet):
                yield a, b
                return
            for (k1, v1), (k2, v2) in itertools.product(cast(a), cast(b)):
                if k1.compatible_with(k2):
                    yield from _pair_recursive(v1, v2)

        yield from _pair_recursive(buckets_a, buckets_b)

    @staticmethod
    def pair_matching_buckets_query(
        buckets_a: dict[fzs[Compatibility], list["InterchangeableSet"]],
        buckets_b: dict[fzs[Compatibility], list["InterchangeableSet"]],
    ):
        for k, v in buckets_a.items():
            if k in buckets_b:
                for v1, v2 in itertools.product(v, buckets_b[k]):
                    yield v1, v2

    @staticmethod
    def call_on_buckets(
        buckets: dict[fzs[Compatibility], list["InterchangeableSet"] | dict],
        func,
    ):
        def _call_recursive(b):
            if isinstance(b, dict):
                for k, v in b.items():
                    b[k] = _call_recursive(v)
                return b
            return func(b)

        _call_recursive(buckets)

    def _hash_keys(self):
        return (
            fzs(self.compatibility),
            fzs(self.payload.keys()),
            fzs(self.shared_resources),
        )

    def __hash__(self) -> int:
        return hash(fzs(self.compatibility))

    def __str__(self):
        c = " ".join(f"[{c}]" for c in sorted(self.compatibility))
        payload_keys = sorted(self.payload.keys())
        return c + " " + " ".join(f"[{k}]" for k in payload_keys)

    def __eq__(self, value: object) -> bool:
        return fzs(self.compatibility) == fzs(value.compatibility)

    def __lt__(self, other: "InterchangeableSet") -> bool:
        return tuple(sorted(self.compatibility)) < tuple(sorted(other.compatibility))

    def __repr__(self):
        return f"InterchangeableSet({self.compatibility})"


class TestInterchangeableSet(unittest.TestCase):
    def test_vertical_combine(self):
        fs = []
        for i in range(2):
            comp = Compatibility(
                einsum_id=f"einsum1",
                tiling={},
            )
            fs.append(InterchangeableSet({comp}, Pareto.get_dummy()))
        new_fs = InterchangeableSet.vertical_combine(fs)
        self.assertEqual(len(new_fs.compatibility), 1)

    def test_combine(self):
        comp1 = Compatibility(
            einsum_id=f"einsum1",
            tiling={"Q": TensorTiling("GLB", (Loop(fzs("A"), 1, False),))},
        )
        comp2 = Compatibility(
            einsum_id=f"einsum2",
            tiling={"V": TensorTiling("GLB", (Loop(fzs("A"), 1, False),))},
        )
        fs1 = InterchangeableSet({comp1}, Pareto.get_dummy())
        fs2 = InterchangeableSet({comp2}, Pareto.get_dummy())
        new_fs = fs1.combine(fs2)
        self.assertEqual(len(new_fs.compatibility), 2)
        self.assertIn(comp1, new_fs.compatibility)
        self.assertIn(comp2, new_fs.compatibility)
        self.assertEqual(new_fs.tensors, {"Q", "V"})

    def test_compatibile_with(self):
        def get_tiling(rank_size: int):
            return {"T": TensorTiling("GLB", (Loop(fzs("A"), rank_size, False),))}

        comp1 = Compatibility(einsum_id="A", tiling=get_tiling(1))
        comp2 = Compatibility(einsum_id="B", tiling=get_tiling(1))

        comp4 = Compatibility(einsum_id="C", tiling=get_tiling(1))
        comp5 = Compatibility(einsum_id="C", tiling=get_tiling(2))

        fs1 = InterchangeableSet({comp1}, Pareto.get_dummy()).combine(
            InterchangeableSet({comp2}, Pareto.get_dummy())
        )
        fs2 = InterchangeableSet({comp4}, Pareto.get_dummy())
        self.assertEqual(fs1.compatible_with(fs2), True)

        fs1 = InterchangeableSet({comp1}, Pareto.get_dummy())
        fs2 = InterchangeableSet({comp5}, Pareto.get_dummy())
        # Not neighbors --> compatible becuase there's nothing overlapping to check
        self.assertEqual(fs1.compatible_with(fs2), False)

    # Test:
    # - Drop dead
    # - Finding live neighbors
    # -
    def test_drop_dead(self):
        comp1 = Compatibility(
            einsum_id=f"einsum1",
            tiling={"Q": TensorTiling("GLB", (Loop(fzs("A"), 1, False),))},
        )
        comp2 = Compatibility(
            einsum_id=f"einsum2",
            tiling={"V": TensorTiling("GLB", (Loop(fzs("A"), 1, False),))},
        )
        fs = InterchangeableSet({comp1}, Pareto.get_dummy()).combine(
            InterchangeableSet({comp2}, Pareto.get_dummy())
        )
        self.assertEqual(len(fs.compatibility), 2)
        fs.drop_dead({"Q"})
        self.assertEqual(len(fs.compatibility), 1)
        self.assertIn(comp1, fs.compatibility)
        fs.drop_dead(set())
        self.assertEqual(len(fs.compatibility), 0)

    def test_live_partition(self):
        ab = {"AB": TensorTiling("GLB", (Loop(fzs("AB"), 1, False),))}
        bc = {"BC": TensorTiling("GLB", ())}
        cd = {"CD": TensorTiling("GLB", ())}
        de = {"DE": TensorTiling("GLB", (Loop(fzs("DE"), 1, False),))}
        ef = {"EF": TensorTiling("GLB", (Loop(fzs("EF"), 1, False),))}

        a = Compatibility(einsum_id="A", tiling={**ab})
        b = Compatibility(einsum_id="B", tiling={**ab, **bc})
        c = Compatibility(einsum_id="C", tiling={**bc, **cd})
        d = Compatibility(einsum_id="D", tiling={**cd, **de})
        e = Compatibility(einsum_id="E", tiling={**de, **ef})
        f = Compatibility(einsum_id="F", tiling={**ef})

        for live, partition in [
            (("AB",), ("", "A", "B")),
            (("BC",), ("", "A", "B", "C")),
            (("CD",), ("", "C", "D", "E", "F")),
            (("DE",), ("", "D", "E", "F")),
            (("EF",), ("", "D", "E", "F")),
            (("AB", "EF"), ("", "A", "B", "D", "E", "F")),
            (("BC", "EF"), ("A", "B", "C", "D", "E", "F")),
            ((), ("",)),
        ]:
            fs = InterchangeableSet({a}, Pareto.get_dummy())
            for z in (b, c, d, e, f):
                fs = fs.combine(InterchangeableSet({z}, Pareto.get_dummy()))
            fs.drop_dead(set(live))
            ids = tuple(sorted("".join(sorted(p for p in p2)) for p2 in fs.payload))
            partition = list(partition)
            partition = tuple(sorted(partition))
            msg = f"Failed with {live} {partition}, got {ids}"
            self.assertEqual(ids, partition, msg)


if __name__ == "__main__":
    unittest.main()
