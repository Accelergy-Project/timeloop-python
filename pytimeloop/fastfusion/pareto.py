import itertools
import re

# Disable numba. We need user_has_package("numba") to be False
import sys

sys.modules["numba"] = None


from paretoset import paretoset

import pandas as pd


MAPPING = "__Mappings"
OCCUPANCY = "__Occupancy"

_resource_name_nloops_reg = re.compile(r"RESOURCE_(.+)_LEVEL_(-?\d+)")


def col2nameloop(x):
    m = _resource_name_nloops_reg.match(x)
    return (m.group(1), int(m.group(2))) if m is not None else None


def nameloop2col(name, nloops):
    return f"RESOURCE_{name}_LEVEL_{nloops}"


MERGE_SUFFIXES = ["_RIGHT_MERGE", "_LEFT_MERGE"]


def is_merge_col(c):
    return any(c.endswith(s) for s in MERGE_SUFFIXES)


def add_to_col(df, c, c2):
    df[c] = df[c] + df[c2] if c in df else df[c2]


def max_to_col(df, c, c2):
    df[c] = df[[c, c2]].max(axis=1) if c in df else df[c2]


# Above index 0: Freed when Einsum fully terminates
# Above index 1: Freed after each iteration of the outermost loop

# -1 -> global resource
# 0 -> einsum only

# Shared index -1: Sum -1 resources, max everyone below
# Shared index 0: Sum 0 resources, max everyone below


def makepareto(data: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in data.columns if c != MAPPING and not is_merge_col(c)]
    return data[paretoset(data[columns])].reset_index(drop=True)


def free_to_loop_index(data: pd.DataFrame, shared_loop_index: int) -> pd.DataFrame:
    keepcols = []
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            name, nloops = name_nloops
            if nloops > shared_loop_index:
                target = nameloop2col(name_nloops[0], shared_loop_index)
                add_to_col(data, target, c)
                c = target
        if c not in keepcols:
            keepcols.append(c)
    return data[keepcols]


def merge_cross(
    a1: pd.DataFrame,
    a2: pd.DataFrame,
    shared_loop_index: int,  # -1 -> no shared loops, 0 -> outermost...
) -> pd.DataFrame:
    a1 = makepareto(free_to_loop_index(a1, shared_loop_index + 1))
    a2 = makepareto(free_to_loop_index(a2, shared_loop_index + 1))

    df = pd.merge(a1, a2, how="cross", suffixes=MERGE_SUFFIXES)
    shared_columns = set(a1.columns) & set(a2.columns)

    # Add the shared resources from one cascade to the leaves of the other
    src_suffix, dst_suffix = MERGE_SUFFIXES
    for _ in range(2):
        for c in shared_columns:
            if c == MAPPING:
                continue
            # If it's not a resource column, just add it
            if (name_nloops := col2nameloop(c)) is None:
                callfunc = add_to_col
            else:
                _, nloops = name_nloops
                callfunc = add_to_col if nloops <= shared_loop_index else max_to_col
            callfunc(df, c, c + src_suffix)
        src_suffix, dst_suffix = dst_suffix, src_suffix

    df = makepareto(df)

    # Merge mappings
    c0, c1 = MAPPING + MERGE_SUFFIXES[0], MAPPING + MERGE_SUFFIXES[1]
    df[MAPPING] = df.apply(lambda row: {**row[c0], **row[c1]}, axis=1)
    # Assert no duplicate columns
    return df[[c for c in df.columns if not is_merge_col(c)]]


class Pareto:
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = makepareto(data)

    @staticmethod
    def vertical_combine(paretos: list["Pareto"]) -> "Pareto":
        return Pareto(pd.concat([p.data for p in paretos]))

    def merge(self, other: "Pareto", shared_loop_index: int) -> "Pareto":
        return Pareto(merge_cross(self.data, other.data, shared_loop_index))

    @staticmethod
    def get_dummy() -> "Pareto":
        df = pd.DataFrame({OCCUPANCY: [1, 2], MAPPING: [{"A": "A"}] * 2})
        return Pareto(df)

    def free_to_loop_index(self, n: int) -> "Pareto":
        self.data = free_to_loop_index(self.data, n)
        self.data = makepareto(self.data)

    def alloc(self, resource_name: str, size: int, above_loop_index: int):
        n = nameloop2col(resource_name, above_loop_index)
        if n in self.data:
            self.data[n] += size
        else:
            self.data[n] = size

    def copy(self) -> "Pareto":
        return Pareto(self.data.copy())


import unittest


class ParetoTest(unittest.TestCase):
    def test_pareto(self):
        occ_key = nameloop2col("GLB", 5)
        data = pd.DataFrame({"A": [1, 2], occ_key: [2, 1], MAPPING: [{"A": "A"}] * 2})
        Pareto(data)

    def test_vertical_combine(self):
        occ_key = nameloop2col("GLB", 5)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                occ_key: [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                occ_key: [3, 3, 1],
                MAPPING: [{"A": "A"}] * 3,
            }
        )

        p1 = Pareto(data1)
        self.assertEqual(len(p1.data), 2)
        p2 = Pareto(data2)
        self.assertEqual(len(p2.data), 1)
        pd12 = Pareto.vertical_combine([p1, p2])
        self.assertEqual(len(pd12.data), 3)

    def test_merge(self):
        data1 = pd.DataFrame(
            {"A": [1, 3, 3], "B": [3, 1, 3], MAPPING: [{"A": "A"}] * 3}
        )
        data2 = pd.DataFrame(
            {"A": [3, 3, 3], "B": [3, 3, 3], MAPPING: [{"A": "A"}] * 3}
        )
        p = Pareto(data1).merge(Pareto(data2), 0)
        d = p.data
        self.assertEqual(d["A"].tolist(), [4, 6])
        self.assertEqual(d["B"].tolist(), [6, 4])

    def test_merge_shared_resources(self):
        occ_key = nameloop2col("GLB", 4)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                MAPPING: [{"A": "A"}] * 3,
                occ_key: [3, 3, 3],
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
                occ_key: [2, 2, 2],
            }
        )
        p = Pareto(data1).merge(Pareto(data2), 5)
        d = p.data
        self.assertEqual(d["A"].tolist(), [4, 6])
        self.assertEqual(d["B"].tolist(), [6, 4])
        self.assertEqual(d[occ_key].tolist(), [5, 5])

        p2 = Pareto(data1).merge(Pareto(data2), 3)
        d = p2.data
        self.assertEqual(d["A"].tolist(), [4, 6])
        self.assertEqual(d["B"].tolist(), [6, 4])
        self.assertEqual(d[occ_key].tolist(), [3, 3])

    def test_merge_shared_resources_2nloops(self):
        occ_key_1 = nameloop2col("GLB", 0)
        occ_key_2 = nameloop2col("GLB", 1)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                MAPPING: [{"A": "A"}] * 3,
                occ_key_1: [3, 3, 3],
                occ_key_2: [8, 8, 8],
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
                occ_key_1: [4, 4, 4],
                occ_key_2: [6, 6, 6],
            }
        )

        # 0 --> GLOBAL RESOURCE
        # 1 --> Shared with all who share ONE loop

        # occ_key_1       occ_key_1    Level 0 shared
        # for             for          Co-tiled with nloops 1 merge
        # occ_key_2       occ_key_2    Level 1 shared
        # for             for          Co-tiled with nloops 2 merge

        # Untiled fused --> Max everything
        d = Pareto(data1).merge(Pareto(data2), -1).data
        self.assertEqual(d[occ_key_1].tolist(), [11, 11])

        # Tiled at nloops 1 --> Sum everything stored at 0
        d = Pareto(data1).merge(Pareto(data2), 0).data
        self.assertEqual(d[occ_key_1].tolist(), [7, 7])
        self.assertEqual(d[occ_key_2].tolist(), [8, 8])

        # Tiled at nloops 2 --> Sum everything stored at 0 and 1
        d = Pareto(data1).merge(Pareto(data2), 1).data
        self.assertEqual(d[occ_key_1].tolist(), [7, 7])
        self.assertEqual(d[occ_key_2].tolist(), [14, 14])

    def test_free_to_loop_index(self):
        # 0 --> Untiled fused
        occ_key_1 = nameloop2col("GLB", 0)
        occ_key_2 = nameloop2col("GLB", 1)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                MAPPING: [{"A": "A"}] * 3,
                occ_key_1: [3, 3, 3],
                occ_key_2: [8, 8, 8],
            }
        )

        p = Pareto(data1)
        d = p.data
        p.free_to_loop_index(2)
        self.assertEqual(d.columns.tolist(), ["A", "B", MAPPING, occ_key_1, occ_key_2])

        p.free_to_loop_index(0)
        d = p.data
        self.assertEqual(d.columns.tolist(), ["A", "B", MAPPING, occ_key_1])
        self.assertEqual(d[occ_key_1].tolist(), [11, 11])


if __name__ == "__main__":
    unittest.main()
