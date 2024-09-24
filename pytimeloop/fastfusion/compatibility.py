import copy
from dataclasses import dataclass
import unittest
from util import fzs


@dataclass(frozen=True)
class OpCompatibility:
    # Fusion information
    fused_tensors: fzs[str]
    fused_loops: tuple[tuple[str, int]]
    fused_ranks: fzs[str]

    # General information about the operation
    ranks: fzs[str]
    tensors: fzs[str]
    neighbors: fzs[str]
    op_name: str

    def get_relevant_fused_loops(
        self, relevant_ranks: set[str]
    ) -> tuple[tuple[str, int]]:
        for i in range(len(self.fused_loops) - 1, -1, -1):
            if self.fused_loops[i][0] in relevant_ranks:
                return self.fused_loops[: i + 1]
        return ()

    def compatible_with(self, other: "OpCompatibility") -> bool:
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
        mine = self.get_relevant_fused_loops(other.ranks)
        other = other.get_relevant_fused_loops(self.ranks)
        if mine and other:
            # We're only worried about loops up to the innermost shared rank
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
                if l[1] == 0:
                    raise ValueError("Loop bound cannot be zero")
                if i == len(big_tiler) - 1 and s[1] % l[1] != 0:
                    return False

        return True

    def drop_irrelevant(
        self,
        relevant_tensors: set[str] = fzs(),
        relevant_ranks: set[str] = fzs(),
        ignore_rank_sizes: bool = False,
    ) -> "OpCompatibility":
        # Relevant fused loops are all loops that are fused with a
        # relevant rank OR are above a loop with a relevant rank.
        fused_loops = self.get_relevant_fused_loops(relevant_ranks)
        if ignore_rank_sizes:
            fused_loops = tuple((r, 1) for r, _ in fused_loops)

        return OpCompatibility(
            op_name=self.op_name,
            fused_tensors=self.fused_tensors & relevant_tensors,
            fused_loops=fused_loops,
            fused_ranks=fzs(r for r, _ in fused_loops),
            ranks=self.ranks & relevant_ranks,
            tensors=self.tensors & relevant_tensors,
            neighbors=self.neighbors,
        )

    @staticmethod
    def get_co_tiled(
        compats: set["OpCompatibility"],
        ops: set[str] = fzs(),
    ) -> set["OpCompatibility"]:
        # Live are:
        # - All neighbors of an original node
        # - All tiled fused with a live node
        live = set()
        to_check = [c for c in compats if c.op_name in ops]
        while to_check:
            c = to_check.pop()
            live.add(c)
            to_check.extend(
                c2
                for c2 in compats
                if c2.op_name in c.neighbors
                and c2.fused_ranks & c.fused_ranks
                and c2 not in live
                and c2 not in to_check
            )
        return live

    @staticmethod
    def get_live(
        compats: set["OpCompatibility"],
        ops: set[str] = fzs(),
    ) -> set["OpCompatibility"]:
        # Live are:
        # - All neighbors of an original node
        # - All tiled fused with a live node
        live = set()
        to_check = [c for c in compats if c.op_name in ops or ops & c.neighbors]
        while to_check:
            c = to_check.pop()
            live.add(c)
            to_check.extend(
                c2
                for c2 in compats
                if c2.op_name in c.neighbors
                and c2.fused_ranks & c.fused_ranks
                and c2 not in live
                and c2 not in to_check
            )
        return live

    @staticmethod
    def get_tiled_partitions(
        compats: set["OpCompatibility"],
    ) -> list[set["OpCompatibility"]]:
        compats = copy.copy(compats)
        partitions = []
        while compats:
            c = next(iter(compats))
            live = OpCompatibility.get_co_tiled(compats, {c.op_name})
            assert c in live
            partitions.append(live)
            compats -= live
        return partitions

    def __eq__(self, other):
        return (
            self.op_name == other.op_name
            and self.fused_tensors == other.fused_tensors
            and len(self.fused_loops) == len(other.fused_loops)
            and all(
                len(a) == len(b) for a, b in zip(self.fused_loops, other.fused_loops)
            )
        )

    def __lt__(self, other):
        return (
            self.op_name < other.op_name
            or self.fused_tensors < other.fused_tensors
            or self.fused_loops < other.fused_loops
        )

    def __str__(self):
        return (
            f"{self.op_name} ftensors {self.fused_tensors}, floops {self.fused_loops}"
        )

    def __repr__(self):
        return (
            f"OpCompatibility({self.op_name}, {self.fused_tensors}, {self.fused_loops})"
        )


class TestOpCompatibility(unittest.TestCase):
    def test_compatible_with(self):
        loopnests = [
            "A1 B2 C3 D4",
            "A1",
            "A1 B2 C3 D8",
            "A1 B2 C6",
            "A1 B2 C5",
            "B2 A1",
            "Q1 A1 B2 C3 D4",
            "",
        ]

        compatibilities = [
            (1, 1, 1, 0, 0, 0, 0, 1),
            (1, 1, 1, 1, 1, 0, 0, 1),
            (1, 1, 1, 0, 0, 0, 0, 1),
            (0, 1, 0, 1, 0, 0, 0, 1),
            (0, 1, 0, 0, 1, 0, 0, 1),
            (0, 0, 0, 0, 0, 1, 0, 1),
            (0, 0, 0, 0, 0, 0, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1),
        ]

        comps = []
        for i, l in enumerate(loopnests):
            comps.append(
                OpCompatibility(
                    op_name=l,
                    fused_tensors=fzs(["T1"]),
                    fused_loops=tuple((r[0], int(r[1])) for r in l.split(" ") if r),
                    fused_ranks=fzs("ABCD"),
                    ranks=fzs("ABCD"),
                    tensors=fzs(["T1"]),
                    neighbors=fzs(),
                )
            )

        for i, c1 in enumerate(comps):
            for j, c2 in enumerate(comps):
                e = bool(compatibilities[i][j])
                self.assertEqual(
                    c1.compatible_with(c2),
                    e,
                    f"{c1.op_name} <-> {c2.op_name} compatible got {not e}, expected {e}",
                )

    def test_get_tiled_partitions(self):
        loopnests = [
            "A1 B2 C3 D4",
            "A1",
            "",
            "A1 B2 C3 D4",
        ]
        comps = []
        for i, l in enumerate(loopnests):
            comps.append(
                OpCompatibility(
                    op_name=l,
                    fused_tensors=fzs(["T1"]),
                    fused_loops=tuple((r[0], int(r[1])) for r in l.split(" ") if r),
                    fused_ranks=fzs("ABCD"),
                    ranks=fzs("ABCD"),
                    tensors=fzs(["T1"]),
                    neighbors=fzs(),
                )
            )

        partitions = OpCompatibility.get_tiled_partitions(set(comps))
        self.assertEqual(len(partitions), 3)


if __name__ == "__main__":
    unittest.main()
