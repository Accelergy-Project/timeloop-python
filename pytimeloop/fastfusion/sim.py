from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import cached_property
import struct
from typing import Any, Iterable, Optional

import pandas as pd

from .pareto import Pareto, LOGSTRING, nameloop2col
from .util import expfmt, fzs

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
        if not isinstance(other, Loop):
            return False
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
        # return ("S-" if self.is_spatial else "") + f"{self.rank_id}-{self.bound}"
        return f"Loop({self.rank_id.__repr__()}, {self.bound}, {self.is_spatial})"

    def __str__(self):
        return ("S-" if self.is_spatial else "") + f"{self.rank_id}-{self.bound}"

    def pydot_str(self):
        if self.is_spatial:
            return f"S-for R{self.rank_id} size {expfmt(self.bound)}"
        return f"for {self.rank_id} size {expfmt(self.bound)}"

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "Loop":
        return Loop(rank_renaming[self.rank_id], self.bound, self.is_spatial)

    def to_yaml(self):
        return {
            "type": "loop",
            "rank_id": self.rank_id,
            "bound": self.bound,
            "is_spatial": self.is_spatial,
        }

@dataclass(frozen=True, order=True)
class TensorStorage:
    # This order is important. Make sure backing storages are the lowest.
    tensor_id: str
    above_loop_index: int
    backer_id: str
    tile_size: int
    # n_repititions: int = 1

    def __tuple__(self):
        return (self.tensor_id, self.backer_id, self.above_loop_index, self.tile_size)

    @property
    def ts(self):
        return self.tile_size

    def __str__(self):
        return f"[{self.backer_id}] {self.tensor_id} sz {expfmt(self.tile_size)} above {self.above_loop_index}"  # x{expfmt(self.n_repititions)}"

    def __repr__(self):
        return f"TensorStorage({repr(self.tensor_id)}, {self.above_loop_index}, {repr(self.backer_id)}, {self.tile_size})"

    def pydot_str(self):
        return f"[{self.backer_id}] {self.tensor_id} size {expfmt(self.tile_size)}"
        # *{expfmt(self.n_repititions)}={expfmt(self.tile_size)}"# * self.n_repititions)}"

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "TensorStorage":
        return TensorStorage(
            tensor_renaming[self.tensor_id],
            self.above_loop_index,
            self.backer_id,
            self.tile_size,
            # self.n_repititions
        )

    def to_yaml(self):
        return {
            "type": "storage",
            "tensor_id": self.tensor_id,
            "backer_id": self.backer_id,
            "above_loop_index": self.above_loop_index,
            "tile_size": self.tile_size,
        }

    def get_backing_stores(all_tensors: set["TensorStorage"]) -> list["TensorStorage"]:
        id2tensor = defaultdict(lambda: [])
        for t in all_tensors:
            id2tensor[t.tensor_id].append(t)
        return sorted(sorted(v)[0] for v in id2tensor.values())
    
    def __eq__(self, value):
        if not isinstance(value, TensorStorage):
            return False
        for to_check in ["tensor_id", "backer_id", "above_loop_index", "tile_size"]:
            a, b = getattr(self, to_check), getattr(value, to_check)
            if a != "*" and b != "*" and a != b:
                return False
        return True

@dataclass(frozen=True)
class Tiling:
    loops: tuple[Loop, ...]
    tensors: fzs[TensorStorage]
    tags: fzs[Any] = fzs()
    
    def __post_init__(self):
        assert isinstance(self.tensors, frozenset)
        assert isinstance(self.loops, tuple)
        assert isinstance(self.tags, frozenset)

    @cached_property
    def tensor_names(self) -> set[str]:
        return {t.tensor_id for t in self.tensors}

    def get_backing_levels(self) -> dict[str, int]:
        backings = {}
        for t in self.tensors:
            prev = backings.get(t.tensor_id, t.above_loop_index)
            backings[t.tensor_id] = min(prev, t.above_loop_index)
        return backings

    def shared_loop_index(self, live_tensors: set[str]) -> int:
        n = [l for t, l in self.get_backing_levels().items() if t in live_tensors]
        return max(n) - 1 if n else -1

    def __eq__(self, other):
        no_tags = (not self.tags and not other.tags)
        tags_match = any(s == o for s in self.tags for o in other.tags)
        return self.loops == other.loops and (no_tags or tags_match) and self.tensors == other.tensors

    def __len__(self) -> int:
        return len(self.loops)

    def __hash__(self):
        return hash((self.loops, self.tensors, self.tags))

    def clear_dead_tensors(
        self, live_tensors: set[str], keep_loops: bool = False
    ) -> "Tiling":
        loops = (
            self.loops
            if keep_loops
            else self.loops[: self.shared_loop_index(live_tensors) + 1]
        )
        tensors = frozenset(t for t in self.tensors if t.tensor_id in live_tensors)
        return Tiling(loops, tensors, self.tags)

    def __lt__(self, other):
        return (self.loops, self.tensors) < (other.loops, other.tensors)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Tiling({self.loops.__repr__()}, {self.tensors.__repr__()}, {self.tags.__repr__()})"
    
    def merge_next(self, n: "Tiling", live_tensors: set[str]) -> "Tiling":
        tensors = fzs(t for t in (n.tensors | self.tensors) if t.tensor_id in live_tensors)
        shared_loop_index = max(t.shared_loop_index(live_tensors) for t in [self, n])
        return Tiling(n.loops[: shared_loop_index + 1], tensors, n.tags)

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "Tiling":
        return Tiling(
            tuple(l.rename(rank_renaming, tensor_renaming) for l in self.loops),
            fzs(t.rename(rank_renaming, tensor_renaming) for t in self.tensors),
            self.tags,
        )
        
    def matches_permutation(self, permutation: list[str]) -> bool:
        i, j = 0, 0
        while True:
            if i == len(self.loops) and j == len(permutation):
                return True
            if j == len(permutation):
                return False

            # Mismatch!
            if i == len(self.loops) or self.loops[i].rank_id != permutation[j]:
                if permutation[j] != "*":
                    return False
                j += 1
                while i < len(self.loops) and (j == len(permutation) or self.loops[i].rank_id != permutation[j]):
                    i += 1
            else:
                i, j = i + 1, j + 1
                
    def has_tensor(self, *tensors: TensorStorage) -> bool:
        return all(any(t == tensor for t in self.tensors) for tensor in tensors)
    
    def set_tags(self, *new_tags: Any) -> "Tiling":
        return Tiling(self.loops, self.tensors, fzs(new_tags))

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

    def copy(self) -> "SIM":
        return SIM(self.tiling, self.mapping.copy())

    def merge_next(
        self, n: "SIM", live_tensors: set[str], delay: bool = False
    ) -> "SIM":
        shared_loop_index = self.tiling.shared_loop_index(n.tiling.tensor_names)
        tiling = self.tiling.merge_next(n.tiling, live_tensors)
        next_shared_loop_index = tiling.shared_loop_index(live_tensors)
        mapping = self.mapping.merge_next(
            n.mapping, shared_loop_index, live_tensors, delay=delay
        )
        s = SIM(tiling, mapping)
        assert (
            len(tiling.loops) == next_shared_loop_index + 1
        ), f"{self.tiling} {n.tiling} {next_shared_loop_index + 1} -> {tiling} {len(tiling.loops)}"
        s.tensors.update(n.tensors)
        s.tensors.update(self.tensors)
        return s

    def get_shared_loop_index(self, live_tensors: set[str]) -> int:
        live_tensors = list(self.tiling.tensor_names) + [live_tensors]
        return self.tiling.shared_loop_index(live_tensors)

    def consolidate(
        self,
        live_tensors: set[str] = None,
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
    ):
        dead_tensors = set(self.tensors) - (live_tensors or set())
        shared_tensors = shared_tensors or set()
        shared_loop_index = self.tiling.shared_loop_index(shared_tensors | live_tensors)
        for t in dead_tensors:
            t = self.tensors.pop(t)
            self.mapping.alloc(t.backer_id, t.tile_size, t.above_loop_index)
        if live_tensors is None:
            self.mapping.free_to_loop_index(0)
            self.mapping.squish_left_right()
        else:
            # Can free the deepest of:
            # - The shared loop with the next SIM
            # - My deepest loop that hasn't yet been freed
            # if self.tensors:
            #     shared_loop_index = max(shared_loop_index, max(t.above_loop_index for t in self.tensors.values()))
            self.mapping.free_to_loop_index(shared_loop_index + 1, resource2capacity)

    def left_consolidate(
        self,
        live_tensors: set[str] = None,
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
    ):
        shared_tensors = shared_tensors or set()
        shared_loop_index = self.tiling.shared_loop_index(shared_tensors | live_tensors)
        for t in self.tensors.values():
            if (
                t.above_loop_index > shared_loop_index
                or t.tensor_id not in shared_tensors | live_tensors
            ):
                self.mapping.alloc(t.backer_id, t.tile_size, t.above_loop_index)
                self.mapping.add_tensor(t)
        if live_tensors is None:
            self.mapping.free_to_loop_index(0)
            self.mapping.squish_left_right(0)
        else:
            # Can free the deepest of:
            # - The shared loop with the next SIM
            # - My deepest loop that hasn't yet been freed
            # if self.tensors:
            #     shared_loop_index = max(shared_loop_index, max(t.above_loop_index for t in self.tensors.values()))
            self.mapping.free_to_loop_index(shared_loop_index + 1, resource2capacity)
            self.mapping.squish_left_right(shared_loop_index + 1)
        self.mapping.make_pareto()

    def __eq__(self, other):
        return self.tiling == other.tiling and self.tensors == other.tensors

    def __hash__(self):
        return hash((self.tiling, fzs(self.tensors.items())))

    @staticmethod
    def concat(sims: Iterable["SIM"]) -> "SIM":
        sims = list(sims)
        assert len(sims) > 0, "Cannot concat empty list of SIMs"
        s = set(frozenset([(k, v) for k, v in s.tensors.items()]) for s in sims)
        assert len(s) == 1, (
            f"Cannot concat SIMs with different tensors:\n\t" +
            "\n\t".join(str(s2) for s2 in s)
        )
        return SIM(sims[0].tiling, Pareto.concat([s.mapping for s in sims]))

    @staticmethod
    def _group(
        sims: list["SIM"], live_tensors: set[str], keep_loops: bool = False
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        grouped = defaultdict(list)
        for s in sims:
            grouped[
                s.tiling.clear_dead_tensors(live_tensors, keep_loops=keep_loops)
            ].append(s)
        return grouped

    @staticmethod
    def combine_combineable(sims: list["SIM"], live_tensors: set[str]) -> list["SIM"]:
        return [SIM.concat(s) for s in SIM._group(sims, live_tensors).values()]

    @staticmethod
    def filter_by_tensor_storages(
        sims: list["SIM"] | dict[Tiling, Any], tensors: set[str]
    ) -> list["SIM"]:
        def check(tensors_to_check):
            for t in tensors_to_check:
                for t2 in tensors:
                    if t.tensor_id == t2.tensor_id and t != t2:
                        return False
            return True

        tensors = set(tensors)
        if isinstance(sims, list):
            return [s for s in sims if check(s.tiling.tensors)]
        if isinstance(sims, dict):
            return {k: v for k, v in sims.items() if check(k.tensors)}
        raise ValueError(f"Invalid type {type(sims)}")

    def filter_by_mapping_hashes(self, hashes: set[str]) -> Optional["SIM"]:
        self = SIM(self.tiling, self.mapping.filter_by_mapping_hashes(hashes))
        return self if len(self.mapping.data) > 0 else None

    def group_left(
        sims: list["SIM"], live_tensors: set[str], keep_loops: bool = False
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors, keep_loops)

    def group_right(
        sims: list["SIM"], live_tensors: set[str]
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors)
    
    def set_tags(self, *tags: Any) -> "SIM":
        self.tiling = self.tiling.set_tags(*tags)
        
    @property
    def tags(self) -> fzs[Any]:
        return self.tiling.tags


import unittest


class TestSIM(unittest.TestCase):
    def test_reservations(self):
        a0 = TensorStorage("A0", 0, "GLB", 1)
        a1 = TensorStorage("A1", 1, "GLB", 2)
        a2 = TensorStorage("A2", 2, "GLB", 3)
        a1b1 = TensorStorage("A1B1", 2, "GLB", 2)  # Above loop 2 --> share 0, 1
        b0 = TensorStorage("B0", 0, "GLB", 4)
        b1 = TensorStorage("B1", 1, "GLB", 5)
        b2 = TensorStorage("B2", 2, "GLB", 6)
        b0c0 = TensorStorage("B0C0", 1, "GLB", 4)  # Above loop 1 --> share 0
        c0 = TensorStorage("C0", 0, "GLB", 7)
        c1 = TensorStorage("C1", 1, "GLB", 8)
        c2 = TensorStorage("C2", 2, "GLB", 9)
        c1d1 = TensorStorage("C1D1", 2, "GLB", 8)  # Above loop 2 --> share 0, 1
        d0 = TensorStorage("D0", 0, "GLB", 10)
        d1 = TensorStorage("D1", 1, "GLB", 11)
        d2 = TensorStorage("D2", 2, "GLB", 12)
        tensors0 = [a0, a1, a2, a1b1]
        tensors1 = [b0, b1, b2, a1b1, b0c0]
        tensors2 = [c0, c1, c2, b0c0, c1d1]
        tensors3 = [d0, d1, d2, c1d1]
        loops = [("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R3")]

        sims: list[SIM] = []
        for tensors, loops in zip([tensors0, tensors1, tensors2, tensors3], loops):
            tiling = Tiling(tuple(Loop(r, 2, False) for r in loops), fzs(tensors))
            mapping = Pareto(pd.DataFrame({"Energy": [1]}))
            mapping.data[LOGSTRING] = [{}]
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
        a0 = TensorStorage("A0", 0, "GLB", 1)
        a1 = TensorStorage("A1", 1, "GLB", 2)
        a2 = TensorStorage("A2", 2, "GLB", 3)
        a1b1 = TensorStorage("A1B1", 2, "GLB", 2)  # Above loop 2 --> share 0, 1
        b0 = TensorStorage("B0", 0, "GLB", 4)  #
        b1 = TensorStorage("B1", 1, "GLB", 5)  #
        b2 = TensorStorage("B2", 2, "GLB", 6)  #
        b0c0 = TensorStorage("B0C0", 1, "GLB", 4)  #  # Above loop 1 --> share 0
        c0 = TensorStorage("C0", 0, "GLB", 7)  # 7
        c1 = TensorStorage("C1", 1, "GLB", 8)  # 8
        c2 = TensorStorage("C2", 2, "GLB", 9)  # 9
        c1d1 = TensorStorage("C1D1", 2, "GLB", 8)  # 8 # Above loop 2 --> share 0, 1
        d0 = TensorStorage("D0", 0, "GLB", 10)  #
        d1 = TensorStorage("D1", 1, "GLB", 11)  #
        d2 = TensorStorage("D2", 2, "GLB", 12)  #

        tensors0 = [a0, a1, a2, a1b1]
        tensors1 = [b0, b1, b2, a1b1, b0c0]
        tensors2 = [c0, c1, c2, b0c0, c1d1]
        tensors3 = [d0, d1, d2, c1d1]

        loops = [("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R3")]

        sims: list[SIM] = []
        for tensors, loops in zip([tensors0, tensors1, tensors2, tensors3], loops):
            tiling = Tiling(tuple(Loop(r, 2, False) for r in loops), fzs(tensors))
            mapping = Pareto(pd.DataFrame({"Energy": [1]}))
            mapping.data[LOGSTRING] = [{}]
            for t in tensors:
                mapping.alloc(t.backer_id, t.tile_size, t.above_loop_index)
            sims.append(SIM(tiling, mapping))

        sims2 = copy.deepcopy(sims)
        while len(sims) > 1:
            sims[0].merge_next(sims.pop(1))
            sims2[0].merge_next(sims2.pop(1))
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
