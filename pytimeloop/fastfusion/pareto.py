from collections import defaultdict
import itertools
import re

# Disable numba. We need user_has_package("numba") to be False
import sys
from typing import Optional

from joblib import delayed

from pytimeloop.fastfusion.util import fzs

sys.modules["numba"] = None


from paretoset import paretoset

import pandas as pd
import functools

MAPPING = "__Mappings"
OCCUPANCY = "__Occupancy"

_resource_name_nloops_reg = re.compile(r"RESOURCE_(.+?)(?:_LEFT)?_LEVEL_(-?\d+)")

def dict_cached(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# TODO: Make these tuples?

@dict_cached
def col2nameloop(x):
    m = _resource_name_nloops_reg.match(x)
    return (m.group(1), int(m.group(2))) if m is not None else None

@dict_cached
def nameloop2col(name, nloops, left: bool=False):
    if left:
        return f"RESOURCE_{name}_LEFT_LEVEL_{nloops}"
    return f"RESOURCE_{name}_LEVEL_{nloops}"

@dict_cached
def is_left_col(x):
    return "_LEFT_LEVEL_" in x

MERGE_SUFFIXES = ["_RIGHT_MERGE", "_LEFT_MERGE"]

def is_merge_col(c):
    return any(c.endswith(s) for s in MERGE_SUFFIXES)


def add_to_col(df, c, c2):
    if c in df:
        df.loc[:, c] = df[c] + df[c2]
    else:
        df.loc[:, c] = df[c2]


def max_to_col(df, c, c2):
    if c in df:
        df.loc[:, c] = df[[c, c2]].max(axis=1)
    else:
        df.loc[:, c] = df[c2]

# Above index 0: Freed when Einsum fully terminates
# Above index 1: Freed after each iteration of the outermost loop

# -1 -> global resource
# 0 -> einsum only

# Shared index -1: Sum -1 resources, max everyone below
# Shared index 0: Sum 0 resources, max everyone below


def makepareto(data: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in data.columns if c != MAPPING and not is_merge_col(c)]
    # Drop any columns that are all zeros
    for c in list(columns):
        if not data[c].any():
            data = data.drop(columns=[c])
            columns.remove(c)
    # TODO: Check if this is helpful. Right will be added to later, so if it is > than the left,
    # we can overwrite the left with the higher value. May be helpful for pareto pruning.
    for c in columns:
        if is_left_col(c):
            continue
        if (name_nloops := col2nameloop(c)) is not None:
            name, n = name_nloops
            left_equivalent = nameloop2col(name, n, left=True)
            if left_equivalent in columns:
                max_to_col(data, left_equivalent, c)
    if len(data) == 1:
        return data
    return data[paretoset(data[columns])].reset_index(drop=True)

def squish_left_right(data: pd.DataFrame):
    nloops2left = defaultdict(set)
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            if is_left_col(c):
                name, nloops = name_nloops
                nloops2left[nloops].add((c, name))
            
    for n in nloops2left.keys():
        for c, name in nloops2left[n]:
            target = nameloop2col(name, n)
            max_to_col(data, target, c)
            
    keepcols = [c for c in data.columns if not is_left_col(c)]
    return data[keepcols]

def free_to_loop_index(data: pd.DataFrame, shared_loop_index: int) -> pd.DataFrame:
    nloops2left = defaultdict(set)
    nloops2right = defaultdict(set)
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            name, nloops = name_nloops
            target = nloops2left if is_left_col(c) else nloops2right
            target[nloops].add((c, name))

    max_nloops = max(max(nloops2left.keys(), default=-1), max(nloops2right.keys(), default=-1))
    for n in range(max_nloops, shared_loop_index, -1):
        # LEFT data: Max to the same level on the right
        for c, name in nloops2left[n]:
            target = nameloop2col(name, n)
            if target in data:
                max_to_col(data, target, c)
            else:
                data.rename(columns={c: target}, inplace=True)
            nloops2right[n].add((target, name))
        # RIGHT data: Sum to the level below on the right
        for c, name in nloops2right[n]:
            target = nameloop2col(name, n - 1)
            if target in data:
                add_to_col(data, target, c)
            else:
                data.rename(columns={c: target}, inplace=True)
            nloops2right[n - 1].add((target, name))

    keepcols = []
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            name, nloops = name_nloops
            if nloops <= shared_loop_index:
                keepcols.append(c)
        else:
            keepcols.append(c)
                
    return data[keepcols]


def merge_cross(
    left: pd.DataFrame,
    right: pd.DataFrame,
    shared_loop_index: int,  # -1 -> no shared loops, 0 -> outermost...
    next_shared_loop_index: int,
    resource2capacity: dict[str, int],
    as_pareto: bool = False,
) -> pd.DataFrame:
    left = makepareto(free_to_loop_index(left, shared_loop_index + 1))
    right = makepareto(right)

    df = pd.merge(left, right, how="cross", suffixes=MERGE_SUFFIXES)
    shared_columns = set(left.columns) & set(right.columns) - set([MAPPING])

    for c in shared_columns:
        # Unknown loop index -> add
        if (name_nloops := col2nameloop(c)) is None:
            callfunc = add_to_col
            for suffix in MERGE_SUFFIXES:
                callfunc(df, c, c + suffix)
        else:
            name, nloops = name_nloops
            # <= shared_loop_index -> add
            if nloops <= shared_loop_index:
                callfunc = add_to_col
                for suffix in MERGE_SUFFIXES:
                    callfunc(df, c, c + suffix)
            # > shared_loop_index -> create new left column
            else:
                max_to_col(df, nameloop2col(name, nloops, left=True), c + MERGE_SUFFIXES[0])
                max_to_col(df, nameloop2col(name, nloops), c + MERGE_SUFFIXES[1])

    # Pipeline:
    # - Need to share temporal loops up to the spatial loop index
    #   Resources:
    #   - Energy
    #   - PE utilization
    #   - Buf utilization
    #   - Buf accesses (for BW calculation later)

    # - Options:
    #   - Non-pipelined: Sum resources above shared loops, max below.
    #   - Pipelined: Sum resources above shared loops, max below. Sum
    #     PE utilization. Latency is pipeline latency summed.
    #     
    #  *  Can't bake into compatiblity unless we have a notion of left vs.
    #     right pipelined.
    
    
    # PIPELINE CHANGES REQUIRED:
    # - Latency above above loop index (first tile), below (all subsequent tiles)
    # - Tiling includes information for how may be fused:
    #   - Pipelined: Max below latencies, 
    #   - Non-pipelined:
    # Shared resources:
    # - 
    # SEQUENTIAL:
    # - In parallel: Fetch all above-shared-loop resources for all operations
    # - Sequentially: Fetch any below-shared-loop resources for all operations
    # PIPELINE:
    # - In parallel: Fetch all above-shared-loop resources for all operations
    # - Sequentially: Fetch any below-shared-loop resources for the first iteration of all operations
    # - In parallel: Fetch all below-shared-loop resources for all operations in all subsequent iterations

    df = free_to_loop_index(df, next_shared_loop_index + 1)
    for resource, capacity in resource2capacity.items():
        colname = nameloop2col(resource, 0)
        if colname in df:
            if capacity is not None:
                df = df[df[colname] <= capacity]
            del df[colname]
  
    # import time
    # t0 = time.time()
    df2 = makepareto(df)
    # t1 = time.time()
    # if t1 - t0 > 2:
    #     print("Took", t1 - t0, "seconds")
    #     df2 = makepareto(df)
        # df2.to_csv("slow.csv")

    df = df2

    # Merge mappings
    c0, c1 = MAPPING + MERGE_SUFFIXES[0], MAPPING + MERGE_SUFFIXES[1]
    if len(df) > 0:
        df[MAPPING] = df.apply(lambda row: {**row[c0], **row[c1]}, axis=1)
    else:
        df[MAPPING] = []
    # try:
    #     df[MAPPING] = df.apply(lambda row: {**row[c0], **row[c1]}, axis=1)
    # except:
    #     merge_cross(left, right, shared_loop_index, next_shared_loop_index, resource2capacity, as_pareto=True)
    df = df[[c for c in df.columns if not is_merge_col(c)]]

    # Assert no NaNs
    assert not df.isnull().values.any()
    
    return Pareto(df, skip_pareto=True) if as_pareto else df


class Pareto:
    def __init__(self, data: pd.DataFrame, skip_pareto: bool=False):
        self.data: pd.DataFrame = data if skip_pareto else makepareto(data)

    def einsum_ids(self):
        return fzs(self.data[MAPPING].iloc[0].keys())

    @staticmethod
    def concat(paretos: list["Pareto"]) -> "Pareto":
        return Pareto(pd.concat([p.data for p in paretos]).fillna(0))

    def merge(self, other: "Pareto", shared_loop_index: int, next_shared_loop_index: int, resource2capacity: dict[str, int], delay: bool=False) -> "Pareto":
        d = delayed(merge_cross)(self.data, other.data, shared_loop_index, next_shared_loop_index, resource2capacity, as_pareto=True)
        return d if delay else d[0](*d[1], **d[2])

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
    
    def limit_capacity(self, resource2capacity: dict[str, Optional[int]]):
        found = False
        for resource, capacity in resource2capacity.items():
            colname = nameloop2col(resource, 0)
            if colname in self.data:
                if capacity is not None:
                    self.data = self.data[self.data[colname] <= capacity]
                del self.data[colname]
                found = True
        if found:
            self.data = makepareto(self.data)
            
    def squish_left_right(self):
        self.data = squish_left_right(self.data)

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
        pd12 = Pareto.concat([p1, p2])
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
        d = squish_left_right(p2.data)
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
        d = squish_left_right(d)
        self.assertEqual(d[occ_key_1].tolist(), [11, 11])

        # Tiled at nloops 1 --> Sum everything stored at 0
        d = Pareto(data1).merge(Pareto(data2), 0).data
        d = squish_left_right(d)
        self.assertEqual(d[occ_key_1].tolist(), [7, 7])
        self.assertEqual(d[occ_key_2].tolist(), [8, 8])

        # Tiled at nloops 2 --> Sum everything stored at 0 and 1
        d = Pareto(data1).merge(Pareto(data2), 1).data
        d = squish_left_right(d)
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
