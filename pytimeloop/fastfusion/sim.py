from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable

import pandas as pd

from pareto import Pareto, MAPPING, nameloop2col
from util import fzs

# Abstractions:
# 1. Each tensor is stored above some loop index. 0 is the outermost loop, 1 the
#    next-innermost...
# 2. All loops above any shared tensor are co-tiled and must match between SIMs.


@dataclass(frozen=True)
class Loop:
    rank_id: str
    bound: int
    is_spatial: bool

    def __eq__(self, other):
        return (
            self.rank_id == other.rank_id
            and self.bound == other.bound
            and self.is_spatial == other.is_spatial
        )

    def __lt__(self, other):
        return (
            self.rank_id < other.rank_id
            or self.bound < other.bound
            or self.is_spatial < other.is_spatial
        )

    def __hash__(self):
        return hash((self.rank_id, self.bound, self.is_spatial))

    def subtiles(self, other: "Loop") -> bool:
        if self.rank_id != other.rank_id:
            return False
        return other.bound % self.bound == 0

    def __repr__(self):
        return ("S-" if self.is_spatial else "") + f"{self.rank_id}-{self.bound}"

    def __str__(self):
        return ("S-" if self.is_spatial else "") + f"{self.rank_id}-{self.bound}"


@dataclass(frozen=True)
class TensorStorage:
    tensor_id: str
    backer_id: str
    above_loop_index: int
    tile_size: int

    def __lt__(self, other: "TensorStorage"):
        return tuple(self) < tuple(other)

    def __tuple__(self):
        return (self.tensor_id, self.backer_id, self.above_loop_index, self.tile_size)

    @property
    def ts(self):
        return self.tile_size


@dataclass(frozen=True)
class Tiling:
    loops: tuple[Loop, ...]
    tensors: tuple[TensorStorage]

    @cached_property
    def tensor_names(self) -> set[str]:
        return {t.tensor_id for t in self.tensors}

    def shared_loop_index(self, other: "Tiling") -> int:
        shared_tensors = self.tensor_names & other.tensor_names
        n = [t.above_loop_index for t in self.tensors if t.tensor_id in shared_tensors]
        if n:
            return max(n) - 1
        return -1

    def __eq__(self, other):
        return self.loops == other.loops

    def get_relevant(self, ranks: set[str]) -> "Tiling":
        for i in range(len(self.loops) - 1, -1, -1):
            if self.loops[i].rank_id in ranks:
                return Tiling(self.loops[: i + 1])
        return Tiling(())

    def __getitem__(self, key: int) -> Loop:
        return Tiling(self.loops[key])

    def __len__(self) -> int:
        return len(self.loops)

    def __hash__(self):
        return hash(self.loops, sorted(self.tensors))


class SIM:
    def __init__(self, tiling: Tiling, mapping: Pareto):
        self.tilings: list[Tiling] = [tiling]
        self.mappings: list[Pareto] = [mapping]
        self.tensors: dict[str, TensorStorage] = {
            t.tensor_id: t for t in tiling.tensors
        }

    def _free_tensor(self, tensor_id: str):
        self.tensors.pop(tensor_id)

    def merge_next(self, n: "SIM", implied_tensors: set[str]) -> "SIM":
        shared_loop_index = self.tilings[-1].shared_loop_index(n.tilings[0])
        self.tilings.extend(n.tilings)
        self.mappings.extend([m.copy() for m in n.mappings])
        self.tensors.update(n.tensors)

        for t in implied_tensors:
            assert (
                False  # This is only for residuals. WRITE TESTS FOR THE FOLLOWING CODE
            )
            # If the tensor is stored above a shared loop, then it is already in shared
            # storage for the existing mappings. Otherwise, we should allocate it in
            # the new mapping.
            if self.tensors[t].above_loop_index > shared_loop_index:
                for m in self.mappings[-len(n.mappings) :]:
                    m.alloc(
                        t, self.tensors[t].tile_size, self.tensors[t].above_loop_index
                    )

    def consolidate(self, final: bool = False):
        # Can merge mappings that have the same # of co-tiled loops as total loops
        tl = self.tilings
        shared_loop_index = [
            tl[i].shared_loop_index(tl[i + 1]) for i in range(len(tl) - 1)
        ]
        if final:
            shared_loop_index.append(-1)
        i = 0
        while i < len(shared_loop_index) - 1:
            if shared_loop_index[i] >= shared_loop_index[i + 1]:
                assert i == 0 or shared_loop_index[i - 1] < shared_loop_index[i]
                self.tilings.pop(i)
                m0, m1 = self.mappings.pop(i), self.mappings.pop(i)
                self.mappings.insert(i, m0.merge(m1, shared_loop_index.pop(i)))
                i = max(0, i - 1)
            else:
                i += 1
        if final:
            self.mappings[0].free_to_loop_index(-1)

    @staticmethod
    def group(sets: Iterable["SIM"], live_tensors: set[str]) -> list["SIM"]:
        grouped = defaultdict(list)
        for s in sets:
            grouped[tuple(s.tilings)].append(s)
        return list(grouped.values())

    def __eq__(self, other):
        return self.tiling == other.tiling and self.tensors == other.tensors

    def __hash__(self):
        return hash((self.tiling, frozenset(s.tensors.items())))

    def __len__(self):
        return len(self.tilings)


import unittest


class TestSIM(unittest.TestCase):
    def test_reservations(self):
        a0 = TensorStorage("A0", "GLB", 0, 1)
        a1 = TensorStorage("A1", "GLB", 1, 2)
        a2 = TensorStorage("A2", "GLB", 2, 3)
        a1b1 = TensorStorage("A1B1", "GLB", 2, 2)  # Above loop 2 --> share 0, 1
        b0 = TensorStorage("B0", "GLB", 0, 4)
        b1 = TensorStorage("B1", "GLB", 1, 5)
        b2 = TensorStorage("B2", "GLB", 2, 6)
        b0c0 = TensorStorage("B0C0", "GLB", 1, 4)  # Above loop 1 --> share 0
        c0 = TensorStorage("C0", "GLB", 0, 7)
        c1 = TensorStorage("C1", "GLB", 1, 8)
        c2 = TensorStorage("C2", "GLB", 2, 9)
        c1d1 = TensorStorage("C1D1", "GLB", 2, 8)  # Above loop 2 --> share 0, 1
        d0 = TensorStorage("D0", "GLB", 0, 10)
        d1 = TensorStorage("D1", "GLB", 1, 11)
        d2 = TensorStorage("D2", "GLB", 2, 12)
        tensors0 = [a0, a1, a2, a1b1]
        tensors1 = [b0, b1, b2, a1b1, b0c0]
        tensors2 = [c0, c1, c2, b0c0, c1d1]
        tensors3 = [d0, d1, d2, c1d1]
        loops = [("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R3")]

        sims: list[SIM] = []
        for tensors, loops in zip([tensors0, tensors1, tensors2, tensors3], loops):
            tiling = Tiling(tuple(Loop(r, 2, False) for r in loops), tuple(tensors))
            mapping = Pareto(pd.DataFrame({"Energy": [1]}))
            mapping.data[MAPPING] = [{}]
            for t in tensors:
                mapping.alloc(t.backer_id, t.tile_size, t.above_loop_index)
            sims.append(SIM(tiling, mapping))

        expected = [
            (1, 2, 5),
            (4, 9, 8),
            (7, 12, 17),
            (10, 11, 20),
        ]
        for e, s in zip(expected, sims):
            for i, e in enumerate(e):
                self.assertEqual(s.mappings[0].data[nameloop2col("GLB", i)].sum(), e)

    def test_all(self):

        a0 = TensorStorage("A0", "GLB", 0, 1)
        a1 = TensorStorage("A1", "GLB", 1, 2)
        a2 = TensorStorage("A2", "GLB", 2, 3)
        a1b1 = TensorStorage("A1B1", "GLB", 2, 2)  # Above loop 2 --> share 0, 1
        b0 = TensorStorage("B0", "GLB", 0, 4)
        b1 = TensorStorage("B1", "GLB", 1, 5)
        b2 = TensorStorage("B2", "GLB", 2, 6)
        b0c0 = TensorStorage("B0C0", "GLB", 1, 4)  # Above loop 1 --> share 0
        c0 = TensorStorage("C0", "GLB", 0, 7)
        c1 = TensorStorage("C1", "GLB", 1, 8)
        c2 = TensorStorage("C2", "GLB", 2, 9)
        c1d1 = TensorStorage("C1D1", "GLB", 2, 8)  # Above loop 2 --> share 0, 1
        d0 = TensorStorage("D0", "GLB", 0, 10)
        d1 = TensorStorage("D1", "GLB", 1, 11)
        d2 = TensorStorage("D2", "GLB", 2, 12)

        tensors0 = [a0, a1, a2, a1b1]
        tensors1 = [b0, b1, b2, a1b1, b0c0]
        tensors2 = [c0, c1, c2, b0c0, c1d1]
        tensors3 = [d0, d1, d2, c1d1]

        loops = [("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R3")]

        sims: list[SIM] = []
        for tensors, loops in zip([tensors0, tensors1, tensors2, tensors3], loops):
            tiling = Tiling(tuple(Loop(r, 2, False) for r in loops), tuple(tensors))
            mapping = Pareto(pd.DataFrame({"Energy": [1]}))
            mapping.data[MAPPING] = [{}]
            for t in tensors:
                mapping.alloc(t.backer_id, t.tile_size, t.above_loop_index)
            sims.append(SIM(tiling, mapping))

        sims2 = copy.deepcopy(sims)
        while len(sims) > 1:
            sims[0].merge_next(sims.pop(1), set())
            sims2[0].merge_next(sims2.pop(1), set())
            sims2[0].consolidate()
        sims[0].consolidate(final=True)
        sims2[0].consolidate(final=True)
        data0 = sims[0].mappings[0].data
        data1 = sims2[0].mappings[0].data
        for k in data0:
            self.assertTrue((data0[k] == data1[k]).all())

        self.assertEqual(len(sims[0]), 1)
        self.assertEqual(len(sims2[0]), 1)
        self.assertEqual(len(sims2[0].mappings), 1)
        self.assertEqual(len(sims2[0].mappings), 1)
        self.assertEqual(len(sims2[0].tilings), 1)
        self.assertEqual(len(sims2[0].tilings), 1)

        expected_util = max(
            max(a2.ts, b2.ts) + a1.ts + b1.ts + a1b1.ts,
            max(c2.ts, d2.ts) + c1.ts + d1.ts + c1d1.ts,
        ) + (b0c0.ts + a0.ts + b0.ts + c0.ts + d0.ts)

        colname = nameloop2col("GLB", -1)
        self.assertEqual(sims[0].mappings[0].data[colname].sum(), expected_util)
        self.assertEqual(sims2[0].mappings[0].data[colname].sum(), expected_util)


if __name__ == "__main__":
    unittest.main()