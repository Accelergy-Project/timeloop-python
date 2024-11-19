from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Optional

import pandas as pd

from .pareto import Pareto, MAPPING, nameloop2col
from .util import fzs

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
        return self.__tuple__() < other.__tuple__()

    def __tuple__(self):
        return (self.tensor_id, self.backer_id, self.above_loop_index, self.tile_size)

    @property
    def ts(self):
        return self.tile_size

    def __str__(self):
        return f"{self.tensor_id}({self.backer_id},{self.above_loop_index})"

    def __repr__(self):
        return f"{self.tensor_id}({self.backer_id},{self.above_loop_index})"


@dataclass(frozen=True)
class Tiling:
    loops: tuple[Loop, ...]
    tensors: fzs[TensorStorage]

    @cached_property
    def tensor_names(self) -> set[str]:
        return {t.tensor_id for t in self.tensors}

    def shared_loop_index(self, live_tensors: set[str]) -> int:
        n = [t.above_loop_index for t in self.tensors if t.tensor_id in live_tensors]
        return max(n) - 1 if n else -1

    def __eq__(self, other):
        return self.loops == other.loops

    def get_relevant(self, ranks: set[str]) -> "Tiling":
        for i in range(len(self.loops) - 1, -1, -1):
            if self.loops[i].rank_id in ranks:
                return Tiling(self.loops[: i + 1])
        return Tiling(())

    def __len__(self) -> int:
        return len(self.loops)

    def __hash__(self):
        return hash((self.loops, self.tensors))

    def clear_dead_tensors(self, live_tensors: set[str]) -> "Tiling":
        return Tiling(
            self.loops[: self.shared_loop_index(live_tensors) + 1],
            tuple(t for t in self.tensors if t.tensor_id in live_tensors),
        )

    def __lt__(self, other):
        return self.loops < other.loops
    
    def __str__(self):
        return "Tiling(loops=" + ", ".join(str(l) for l in self.loops) + \
            ", tensors=" + ", ".join(str(t) for t in sorted(self.tensors)) + ")"
    
    def __repr__(self):
        return self.__str__()

class SIM:
    def __init__(self, tiling: Tiling, mapping: Pareto):
        self.tiling: Tiling = tiling
        self.mapping: Pareto = mapping
        self.tensors: dict[str, TensorStorage] = {
            t.tensor_id: t for t in self.tiling.tensors
        }

    def tiling_str(self):
        tiling = ",".join(str(l) for l in self.tiling.loops)
        tiling += " || " + ", ".join(str(t) for t in self.tensors.values())
        return tiling

    def mapping_str(self):
        return str(self.mapping.einsum_ids())

    @cached_property
    def tensor_names(self) -> set[str]:
        return set(self.tensors)

    def _free_tensor(self, tensor_id: str):
        self.tensors.pop(tensor_id)

    def copy(self) -> "SIM":
        return SIM(self.tiling, self.mapping.copy())

    def merge_next(self, n: "SIM", implied_tensors: set[str], delay: bool) -> "SIM":
        shared_loop_index = self.tiling.shared_loop_index(n.tiling.tensor_names)
        tiling = n.tiling
        mapping = self.mapping.merge(n.mapping, shared_loop_index, delay=delay)
        s = SIM(tiling, mapping)
        s.tensors.update(n.tensors)
        s.tensors.update(self.tensors)
        assert not implied_tensors, "TODO"
        return s
        
        self.tiling = n.tiling
        # TODO: This copy() may not be needed because we squish together left mapping,
        # so each right mapping will be merged with only one?
        self.mapping = self.mapping.merge(n.mapping, shared_loop_index, delay=delay)
        self.tensors.update(n.tensors)

        for t in implied_tensors:
            assert (
                False  # This is only for residuals. WRITE TESTS FOR THE FOLLOWING CODE
            )
            # If the tensor is stored above a shared loop, then it is already in shared
            # storage for the existing mapping. Otherwise, we should allocate it in
            # the new mapping.
            if self.tensors[t].above_loop_index > shared_loop_index:
                for m in self.mapping[-len(n.mapping) :]:
                    m.alloc(
                        t, self.tensors[t].tile_size, self.tensors[t].above_loop_index
                    )
                    
    def _limit_capacity(self, resource2capacity: dict[str, int]):
        if resource2capacity is None:
            return
        self.mapping.limit_capacity(resource2capacity)

    def get_shared_loop_index(self, next_live_tensors: set[str]) -> int:
        live_tensors = list(self.tiling.tensor_names) + [next_live_tensors]
        return self.tiling.shared_loop_index(live_tensors)

    def consolidate(self, next_live_tensors: set[str] = None, resource2capacity: dict[str, int] = None):
        if next_live_tensors is None:
            self.mapping.free_to_loop_index(0)
            self.mapping.squish_left_right()
        self._limit_capacity(resource2capacity)
        return
        if len(self) <= 1:
            self._limit_capacity(resource2capacity)
            return

        # Can merge mapping that have the same # of co-tiled loops as total loops
        tl = self.tiling
        live_tensors = [t.tensor_names for t in tl] + [next_live_tensors]
        shared_loop_index = [
            tl[i].shared_loop_index(live_tensors[i + 1]) for i in range(len(tl))
        ]
        i = 0
        while i < len(shared_loop_index) - 1:
            if shared_loop_index[i] >= shared_loop_index[i + 1]:
                assert i == 0 or shared_loop_index[i - 1] < shared_loop_index[i]
                self.tiling.pop(i)
                m0, m1 = self.mapping.pop(i), self.mapping.pop(i)
                shared_index = shared_loop_index.pop(i)
                m0.free_to_loop_index(shared_index+1)
                m1.free_to_loop_index(shared_index+1)
                self.mapping.insert(i, m0.merge(m1, shared_index))
                self._limit_capacity(resource2capacity, i)
                i = max(0, i - 1)
            else:
                i += 1
        if len(self.mapping) == 1:
            self.mapping.free_to_loop_index(shared_loop_index[0]+1)
        self._limit_capacity(resource2capacity)

    def clear_dead_tensors(self, live_tensors: set[str]):
        dead_tensors = set(self.tensors) - live_tensors
        for t in dead_tensors:
            self._free_tensor(t)
        self.tiling = self.tiling.clear_dead_tensors(live_tensors)

    def __eq__(self, other):
        return self.tiling == other.tiling and self.tensors == other.tensors

    def __hash__(self):
        return hash((self.tiling, fzs(self.tensors.items())))

    @staticmethod
    def concat(sims: Iterable["SIM"]) -> "SIM":
        sims = list(sims)
        assert len(sims) > 0, "Cannot concat empty list of SIMs"
        return SIM(sims[0].tiling, Pareto.concat([s.mapping for s in sims]))

    @staticmethod
    def _group(sims: list["SIM"], live_tensors: set[str]) -> dict[tuple[Tiling, ...], list["SIM"]]:
        grouped = defaultdict(list)
        for s in sims:
            grouped[s.tiling.clear_dead_tensors(live_tensors)].append(s)
        return grouped

    @staticmethod
    def combine_combineable(sims: list["SIM"], live_tensors: set[str]) -> list["SIM"]:
        return [SIM.concat(s) for s in SIM._group(sims, live_tensors).values()]

    def group_by_right(
        sims: list["SIM"], live_tensors: set[str]
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors)

    def group_by_left(
        sims: list["SIM"], live_tensors: set[str]
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors)


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
            tiling = Tiling(tuple(Loop(r, 2, False) for r in loops), fzs(tensors))
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
                self.assertEqual(s.mapping.data[nameloop2col("GLB", i)].sum(), e)

    def test_all(self):
        a0 = TensorStorage("A0", "GLB", 0, 1)
        a1 = TensorStorage("A1", "GLB", 1, 2)
        a2 = TensorStorage("A2", "GLB", 2, 3)
        a1b1 = TensorStorage("A1B1", "GLB", 2, 2)  # Above loop 2 --> share 0, 1
        b0 = TensorStorage("B0", "GLB", 0, 4) # 
        b1 = TensorStorage("B1", "GLB", 1, 5) # 
        b2 = TensorStorage("B2", "GLB", 2, 6) # 
        b0c0 = TensorStorage("B0C0", "GLB", 1, 4) #  # Above loop 1 --> share 0
        c0 = TensorStorage("C0", "GLB", 0, 7) # 7
        c1 = TensorStorage("C1", "GLB", 1, 8) # 8
        c2 = TensorStorage("C2", "GLB", 2, 9) # 9
        c1d1 = TensorStorage("C1D1", "GLB", 2, 8) # 8 # Above loop 2 --> share 0, 1
        d0 = TensorStorage("D0", "GLB", 0, 10) # 
        d1 = TensorStorage("D1", "GLB", 1, 11) # 
        d2 = TensorStorage("D2", "GLB", 2, 12) # 

        tensors0 = [a0, a1, a2, a1b1]
        tensors1 = [b0, b1, b2, a1b1, b0c0]
        tensors2 = [c0, c1, c2, b0c0, c1d1]
        tensors3 = [d0, d1, d2, c1d1]

        loops = [("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R3")]

        sims: list[SIM] = []
        for tensors, loops in zip([tensors0, tensors1, tensors2, tensors3], loops):
            tiling = Tiling(tuple(Loop(r, 2, False) for r in loops), fzs(tensors))
            mapping = Pareto(pd.DataFrame({"Energy": [1]}))
            mapping.data[MAPPING] = [{}]
            for t in tensors:
                mapping.alloc(t.backer_id, t.tile_size, t.above_loop_index)
            sims.append(SIM(tiling, mapping))

        sims2 = copy.deepcopy(sims)
        while len(sims) > 1:
            sims[0].merge_next(sims.pop(1), set())
            sims2[0].merge_next(sims2.pop(1), set())
            sims2[0].consolidate(set().union(*[s.tensor_names for s in sims2]))
        sims[0].consolidate()
        sims2[0].consolidate()
        data0 = sims[0].mapping.data
        data1 = sims2[0].mapping.data
        for k in data0:
            self.assertTrue((data0[k] == data1[k]).all())

        expected_util = max(
            max(a2.ts, b2.ts) + a1.ts + b1.ts + a1b1.ts,
            max(c2.ts, d2.ts) + c1.ts + d1.ts + c1d1.ts,
        ) + (b0c0.ts + a0.ts + b0.ts + c0.ts + d0.ts)

        colname = nameloop2col("GLB", 0)
        self.assertEqual(sims[0].mapping.data[colname].sum(), expected_util)
        self.assertEqual(sims2[0].mapping.data[colname].sum(), expected_util)


if __name__ == "__main__":
    unittest.main()
