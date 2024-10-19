import itertools

# Disable numba. We need user_has_package("numba") to be False
# import sys
# sys.modules["numba"] = None


from paretoset import paretoset

import pandas as pd
from dataclasses import dataclass
from compatibility import SharedResource

from util import fzs


MAPPING = "__Mappings"
OCCUPANCY = f"__Occupancy"


MERGE_SUFFIXES = ["_RIGHT_MERGE", "_LEFT_MERGE"]


COMBINE_FUNCTIONS = {
    OCCUPANCY: lambda df, c: suffix_access(df, c).max(axis=1),
    "default": lambda df, c: suffix_access(df, c).sum(axis=1),
}


def suffix_access(d, c):
    return d[[c + MERGE_SUFFIXES[0], c + MERGE_SUFFIXES[1]]]


def suffix_access_tuple(d, c):
    return d[c + MERGE_SUFFIXES[0]], d[c + MERGE_SUFFIXES[1]]


def merge_cross(a1: pd.DataFrame, a2: pd.DataFrame) -> pd.DataFrame:
    # Before merging, subtract all shared tile size. We'll add it back later
    d = pd.merge(a1, a2, how="cross", suffixes=MERGE_SUFFIXES)

    for c in a2.columns:
        if c == OCCUPANCY or not c.startswith("__"):
            f = COMBINE_FUNCTIONS.get(c, COMBINE_FUNCTIONS.get("default"))
            d[c] = f(d, c)

    d = Pareto.pareto(d)
    s = suffix_access(d, MAPPING)
    c0, c1 = s.columns
    d[MAPPING] = s.apply(lambda row: {**row[c0], **row[c1]}, axis=1)
    d = d[[c for c in d.columns if not any(c.endswith(s) for s in MERGE_SUFFIXES)]]
    return d


class Pareto:
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = Pareto.pareto(data)

    @staticmethod
    def pareto(data: pd.DataFrame) -> pd.DataFrame:
        if len(data) <= 1:
            return data
        d = data[[c for c in data.columns if c == OCCUPANCY or not c.startswith("__")]]
        return data[paretoset(d)].reset_index(drop=True)

    @staticmethod
    def vertical_combine(paretos: list["Pareto"]) -> "Pareto":
        return Pareto(pd.concat([p.data for p in paretos]))

    def combine(self, other: "Pareto") -> "Pareto":
        d0, d1 = self.data, other.data
        d = merge_cross(d0, d1)
        return Pareto(d)

    @staticmethod
    def combine_all(paretos: list["Pareto"]) -> "Pareto":
        p = None
        for p2 in paretos:
            p = p.combine(p2) if p is not None else p2
        return p

    def add_shared_resource(self, resource: SharedResource) -> "Pareto":
        d = self.data.copy()
        for k, v in resource.data:
            d[k] += v
        return Pareto(d)

    @staticmethod
    def get_dummy() -> "Pareto":
        df = pd.DataFrame({OCCUPANCY: [1, 2], MAPPING: [{"A": "A"}] * 2})
        return Pareto(df)


import unittest


class ParetoTest(unittest.TestCase):
    def test_pareto(self):
        data = pd.DataFrame({"A": [1, 2], OCCUPANCY: [2, 1], MAPPING: [{"A": "A"}] * 2})
        Pareto(data)

    def test_vertical_combine(self):
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                OCCUPANCY: [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                OCCUPANCY: [3, 3, 1],
                MAPPING: [{"A": "A"}] * 3,
            }
        )

        p1 = Pareto(data1)
        self.assertEqual(len(p1.data), 2)
        p2 = Pareto(data2)
        self.assertEqual(len(p2.data), 1)

        pd12 = Pareto.vertical_combine([p1, p2])
        self.assertEqual(len(pd12.data), 3)

    def test_combine(self):
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                OCCUPANCY: [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                OCCUPANCY: [3, 3, 1],
                MAPPING: [{"B": "B"}] * 3,
            }
        )
        p = Pareto(data1).combine(Pareto(data2))
        self.assertEqual(len(p.data), 2)

        d = p.data
        # Column "A" should be 4, 6
        self.assertEqual(d["A"].tolist(), [4, 6])
        # Column "B" should be 6, 4
        self.assertEqual(d["B"].tolist(), [6, 4])
        # Column UTIL should be 3, 3
        self.assertEqual(d[OCCUPANCY].tolist(), [3, 3])

    def test_add_shared_resource(self):
        data = pd.DataFrame(
            {
                "A": [1, 3],
                "B": [3, 1],
                OCCUPANCY: [3, 3],
                MAPPING: [{"A": "A"}] * 2,
            }
        )
        p = Pareto(data)
        r = SharedResource("A", frozenset([("A", 1)]), 1)
        p2 = p.add_shared_resource(r)
        self.assertEqual(p2.data[OCCUPANCY].tolist(), [3, 3])
        self.assertEqual(p2.data["A"].tolist(), [2, 4])


if __name__ == "__main__":
    unittest.main()
