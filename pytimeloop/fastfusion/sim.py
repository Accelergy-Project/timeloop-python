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

    def rename(self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]) -> "Loop":
        return Loop(rank_renaming[self.rank_id], self.bound, self.is_spatial)
    
    def to_yaml(self):
        return {
            "type": "loop",
            "rank_id": self.rank_id,
            "bound": self.bound,
            "is_spatial": self.is_spatial,
        }

@dataclass(frozen=True)
class TensorStorage:
    tensor_id: str
    backer_id: str
    above_loop_index: int
    tile_size: int
    n_repititions: int = 1

    def __lt__(self, other: "TensorStorage"):
        return self.__tuple__() < other.__tuple__()

    def __tuple__(self):
        return (self.tensor_id, self.backer_id, self.above_loop_index, self.tile_size)

    @property
    def ts(self):
        return self.tile_size

    def __str__(self):
        return f"{self.tensor_id} sz {expfmt(self.tile_size)} in {self.backer_id} above {self.above_loop_index} x{expfmt(self.n_repititions)}"

    def __repr__(self):
        return self.__str__()
    
    def pydot_str(self):
        return f"{self.tensor_id} in {self.backer_id} size " \
            f"{expfmt(self.tile_size)}*{expfmt(self.n_repititions)}={expfmt(self.tile_size * self.n_repititions)}"
    
    def rename(self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]) -> "TensorStorage":
        return TensorStorage(
            tensor_renaming[self.tensor_id], 
            self.backer_id, 
            self.above_loop_index, 
            self.tile_size, 
            self.n_repititions
        )
    
    def to_yaml(self):
        return {
            "type": "storage",
            "tensor_id": self.tensor_id,
            "backer_id": self.backer_id,
            "above_loop_index": self.above_loop_index,
            "tile_size": self.tile_size,
        }


@dataclass(frozen=True)
class Tiling:
    loops: tuple[Loop, ...]
    tensors: fzs[TensorStorage]

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

    def clear_dead_tensors(
        self, 
        live_tensors: set[str],
        keep_loops: bool = False
    ) -> "Tiling":
        loops = self.loops if keep_loops else self.loops[: self.shared_loop_index(live_tensors) + 1]
        tensors = tuple(t for t in self.tensors if t.tensor_id in live_tensors)
        return Tiling(loops, tensors)

    def __lt__(self, other):
        return self.loops < other.loops
    
    def __str__(self):
        return "Tiling(loops=" + ", ".join(str(l) for l in self.loops) + \
            ", tensors=" + ", ".join(str(t) for t in sorted(self.tensors)) + ")"
    
    def __repr__(self):
        return self.__str__()
    
    def absorb_tensors(self, prev: "Tiling", live_tensors: set[str]) -> "Tiling":
        tensors = fzs(t for t in (prev.tensors | self.tensors) if t.tensor_id in live_tensors)
        shared_loop_index = max(t.shared_loop_index(live_tensors) for t in [self, prev])
        return Tiling(self.loops[:shared_loop_index+1], tensors)
    
    def rename(self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]) -> "Tiling":
        return Tiling(
            tuple(l.rename(rank_renaming, tensor_renaming) for l in self.loops), 
            fzs(t.rename(rank_renaming, tensor_renaming) for t in self.tensors)
        )

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
        t = self.tensors.pop(tensor_id)
        self.mapping.alloc(t.backer_id, t.tile_size, t.above_loop_index)

    def copy(self) -> "SIM":
        return SIM(self.tiling, self.mapping.copy())

    def merge_next(self, n: "SIM", live_tensors: set[str], resource2capacity: dict[str, int], delay: bool=False) -> "SIM":
        shared_loop_index = self.tiling.shared_loop_index(n.tiling.tensor_names)
        tiling = n.tiling.absorb_tensors(self.tiling, live_tensors)
        next_shared_loop_index = tiling.shared_loop_index(live_tensors)
        mapping = self.mapping.merge(n.mapping, shared_loop_index, next_shared_loop_index, resource2capacity, delay=delay)
        s = SIM(tiling, mapping)
        assert len(tiling.loops) == next_shared_loop_index + 1, f"{self.tiling} {n.tiling} {next_shared_loop_index + 1} -> {tiling} {len(tiling.loops)}"
        s.tensors.update(n.tensors)
        s.tensors.update(self.tensors)
        return s

    def get_shared_loop_index(self, next_live_tensors: set[str]) -> int:
        live_tensors = list(self.tiling.tensor_names) + [next_live_tensors]
        return self.tiling.shared_loop_index(live_tensors)

    def consolidate(self, next_live_tensors: set[str] = None, resource2capacity: dict[str, int] = None):
        dead_tensors = set(self.tensors) - (next_live_tensors or set())
        for t in dead_tensors:
            self._free_tensor(t)
        if next_live_tensors is None:
            self.mapping.free_to_loop_index(0)
            self.mapping.squish_left_right()
        else:
            # Can free the deepest of:
            # - The shared loop with the next SIM
            # - My deepest loop that hasn't yet been freed
            shared_loop_index = self.tiling.shared_loop_index(next_live_tensors)
            if self.tensors:
                shared_loop_index = max(shared_loop_index, max(t.above_loop_index for t in self.tensors.values()))
            self.mapping.free_to_loop_index(shared_loop_index+1, resource2capacity)

    def __eq__(self, other):
        return self.tiling == other.tiling and self.tensors == other.tensors

    def __hash__(self):
        return hash((self.tiling, fzs(self.tensors.items())))

    @staticmethod
    def concat(sims: Iterable["SIM"]) -> "SIM":
        sims = list(sims)
        assert len(sims) > 0, "Cannot concat empty list of SIMs"
        assert len(set(frozenset([(k, v) for k, v in s.tensors.items()]) for s in sims)) == 1, "Cannot concat SIMs with different tensors"
        return SIM(sims[0].tiling, Pareto.concat([s.mapping for s in sims]))

    @staticmethod
    def _group(sims: list["SIM"], live_tensors: set[str], keep_loops: bool = False) -> dict[tuple[Tiling, ...], list["SIM"]]:
        grouped = defaultdict(list)
        for s in sims:
            grouped[s.tiling.clear_dead_tensors(live_tensors, keep_loops=keep_loops)].append(s)
        return grouped

    @staticmethod
    def combine_combineable(sims: list["SIM"], live_tensors: set[str]) -> list["SIM"]:
        return [SIM.concat(s) for s in SIM._group(sims, live_tensors).values()]

    def group_by_right(
        sims: list["SIM"], live_tensors: set[str], keep_loops: bool = False
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors, keep_loops)

    def group_by_left(
        sims: list["SIM"], live_tensors: set[str]
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors)

    # def to_pydot_acc_util(self, title: str, info_text: pd.DataFrame):
    #     import pydot
        
    #     title = f"{title}\n" + pretty_sci_notation(
    #         f"{info_text[paretos.UTIL_COL]:.2e} utilization (LEFT), "
    #         f"{info_text[paretos.ACCESSES_COL]:.2e} accesses (RIGHT)"
    #     )

    #     def getstat(s):
    #         return info_text.get(s, 0)

    #     max_op_util = max(getstat(f"{op.name} {paretos.UTIL_COL}") for op in self)
    #     max_op_acc = max(getstat(f"{op.name} {paretos.ACCESSES_COL}") for op in self)
    #     max_t_util = max(getstat(f"{t.name} {paretos.UTIL_COL}") for t in self.tensors)
    #     max_t_acc = max(
    #         getstat(f"{t.name} {paretos.ACCESSES_COL}") for t in self.tensors
    #     )

    #     def get_tensor_node(tensor, acc_info: bool):
    #         acc = getstat(f"{tensor.name} {paretos.ACCESSES_COL}")
    #         util = getstat(f"{tensor.name} {paretos.UTIL_COL}")
    #         tsize = tensor.size()

    #         if acc_info:
    #             s, t = acc / max(max_t_acc, 1), f"{acc:.1e}A (x{acc//tsize})"
    #         else:
    #             s, t = (
    #                 util / max(max_t_util, 1),
    #                 f"{util:.1e}U (/{tsize//max(util, 1)})",
    #             )

    #         t = pretty_sci_notation(t)

    #         importance = 0.3 + 0.7 * s
    #         basecolor = struct.unpack("BBB", bytes.fromhex("f9f9f9"))
    #         scaled = (int(i * importance) for i in basecolor)

    #         text = f"{tensor.name}\n{t}"
    #         node = pydot.Node(
    #             name=f"Tensor {tensor.name} {acc_info}".replace(":", "_").replace(
    #                 "/", "_"
    #             ),
    #             shape="box",
    #             style="filled",
    #             fillcolor="#" + bytes.hex(struct.pack("BBB", *scaled)),
    #             label=text,
    #             fontsize="10",
    #             width=0,
    #             penwidth=1,
    #             margin="0",
    #             height=0,
    #         )
    #         graph.add_node(node)
    #         tensor._graph_node = node
    #         return node

    #     def get_op_node(op, acc_info: bool):
    #         acc = getstat(f"{op.name} {paretos.ACCESSES_COL}")
    #         util = getstat(f"{op.name} {paretos.UTIL_COL}")

    #         if acc_info:
    #             s, t = acc / max(max_op_acc, 1), f"{acc:.1e}A"
    #         else:
    #             s, t = util / max(max_op_util, 1), f"{util:.1e}U"
    #         t = pretty_sci_notation(t)

    #         importance = 0.3 + 0.7 * s
    #         basecolor = struct.unpack("BBB", bytes.fromhex("ffcccc"))
    #         scaled = (int(i * importance) for i in basecolor)

    #         text = f"{op.name} (x{op.scale_accesses_by})\n{t}"
    #         node = pydot.Node(
    #             f"{op.name} {acc_info}".replace(":", "_").replace("/", "_"),
    #             shape="box",
    #             style="filled",
    #             fillcolor="#" + bytes.hex(struct.pack("BBB", *scaled)),
    #             label=text,
    #             fontsize="10",
    #             width=0,
    #             margin="0",
    #             height=0,
    #         )
    #         graph.add_node(node)
    #         op._graph_node = node
    #         return node

    #     graph = pydot.Dot(
    #         graph_type="digraph",
    #         label=title,
    #         ranksep="0.1",
    #         nodesep="0.1",
    #         labelloc="t",
    #     )

    #     # Create nodes for each tensor and add them to the graph
    #     for tensor in self.tensors:
    #         get_tensor_node(tensor, False)
    #     for op in self:
    #         get_op_node(op, False)
    #         for t in op.input_tensors:
    #             edge = pydot.Edge(t._graph_node, op._graph_node, color="blue")
    #             graph.add_edge(edge)
    #         for t in op.output_tensors:
    #             edge = pydot.Edge(op._graph_node, t._graph_node, color="red")
    #             graph.add_edge(edge)

    #     for tensor in self.tensors:
    #         get_tensor_node(tensor, True)
    #     for op in self:
    #         get_op_node(op, True)
    #         for t in op.input_tensors:
    #             edge = pydot.Edge(t._graph_node, op._graph_node, color="blue")
    #             graph.add_edge(edge)
    #         for t in op.output_tensors:
    #             edge = pydot.Edge(op._graph_node, t._graph_node, color="red")
    #             graph.add_edge(edge)

    #     return graph

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
