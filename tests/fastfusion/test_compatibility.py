import unittest

from pytimeloop.fastfusion.compatibility import *


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
                    einsum_id=l,
                    fused_tensors=fzs(["T1"]),
                    fused_loops=tuple((r[0], int(r[1])) for r in l.split(" ") if r),
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
                    f"{c1.einsum_id} <-> {c2.einsum_id} compatible got {not e}, expected {e}",
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
                    einsum_id=l,
                    fused_tensors=fzs(["T1"]),
                    fused_loops=tuple((r[0], int(r[1])) for r in l.split(" ") if r),
                    ranks=fzs("ABCD"),
                    tensors=fzs(["T1"]),
                    neighbors=fzs(),
                )
            )

        partitions = OpCompatibility.get_tiled_partitions(set(comps))
        self.assertEqual(len(partitions), 3)


if __name__ == "__main__":
    unittest.main()