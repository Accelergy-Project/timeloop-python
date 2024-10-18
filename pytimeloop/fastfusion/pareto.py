import itertools
from paretoset import paretoset
import pandas as pd
from dataclasses import dataclass

from util import fzs


MAPPING = "__Mappings"
OCCUPANCY = f"__Occupancy"


DATA_SIZE = lambda x: f"__{x} Datawidth"
NUM_ELEMS = lambda x: f"__{x} Num Elems"


MERGE_SUFFIXES = ["_RIGHT_MERGE", "_LEFT_MERGE"]


COMBINE_FUNCTION_WITHIN_TILED_FUSED = {
    OCCUPANCY: lambda df, c: suffix_access(df, c).sum(axis=1),
    "default": lambda df, c: suffix_access(df, c).sum(axis=1),
}


COMBINE_FUNCTION_DEFAULT = {
    OCCUPANCY: lambda df, c: suffix_access(df, c).max(axis=1),
    "default": lambda df, c: suffix_access(df, c).sum(axis=1),
}


def suffix_access(d, c):
    return d[[c + MERGE_SUFFIXES[0], c + MERGE_SUFFIXES[1]]]


def suffix_access_tuple(d, c):
    return d[c + MERGE_SUFFIXES[0]], d[c + MERGE_SUFFIXES[1]]


def merge_cross(
    a1: pd.DataFrame,
    a2: pd.DataFrame,
    combine_functions: dict[str, callable],
    tensors: set[str] = fzs(),
) -> pd.DataFrame:
    # Before merging, subtract all shared tile size. We'll add it back later
    for t, x in itertools.product(tensors, [a1, a2]):
        x[OCCUPANCY] -= x[DATA_SIZE(t)] * x[NUM_ELEMS(t)]

    d = pd.merge(a1, a2, how="cross", suffixes=MERGE_SUFFIXES)

    for c in a2.columns:
        if c == OCCUPANCY or not c.startswith("__"):
            f = combine_functions.get(c, combine_functions.get("default"))
            d[c] = f(d, c)

    for t in tensors:
        # Next, add new tile size. Save the larger of the two
        max_data_size = suffix_access(d, DATA_SIZE(t)).max(axis=1)
        max_num_elems = suffix_access(d, NUM_ELEMS(t)).max(axis=1)
        d[DATA_SIZE(t)], d[NUM_ELEMS(t)] = max_data_size, max_num_elems
        d[OCCUPANCY] += max_data_size * max_num_elems

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
        d = data[[c for c in data.columns if c == OCCUPANCY or not c.startswith("__")]]
        return data[paretoset(d)].reset_index(drop=True)

    @staticmethod
    def vertical_combine(paretos: list["Pareto"]) -> "Pareto":
        return Pareto(pd.concat([p.data for p in paretos]))

    def combine_dead(self, other: "Pareto") -> "Pareto":
        d0, d1 = self.data, other.data
        d = merge_cross(d0, d1, COMBINE_FUNCTION_DEFAULT)
        return Pareto(d)

    def combine_live(self, other: "Pareto") -> "Pareto":
        d0, d1 = self.data, other.data
        d = merge_cross(d0, d1, COMBINE_FUNCTION_WITHIN_TILED_FUSED)
        return Pareto(d)

    @staticmethod
    def combine_dead_all(paretos: list["Pareto"]) -> "Pareto":
        p = None
        for p2 in paretos:
            p = p.combine_dead(p2) if p is not None else p2
        return p

    @staticmethod
    def combine_live_all(paretos: list["Pareto"]) -> "Pareto":
        p = None
        for p2 in paretos:
            p = p.combine_live(p2) if p is not None else p2
        return p

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

    # def test_combine(self):
    #     data1 = pd.DataFrame(
    #         {
    #             "A": [1, 3, 3],
    #             "B": [3, 1, 3],
    #             OCCUPANCY: [3, 3, 3],
    #             MAPPING: [{"A": "A"}] * 3,
    #         }
    #     )
    #     data2 = pd.DataFrame(
    #         {
    #             "A": [3, 3, 3],
    #             "B": [3, 3, 3],
    #             OCCUPANCY: [3, 3, 1],
    #             MAPPING: [{"B": "B"}] * 3,
    #         }
    #     )

    #     p1 = Pareto(data1)
    #     p2 = Pareto(data2)

    #     pd12 = p1.combine(p2)
    #     x = pd12.data
    #     self.assertEqual(len(next(x)), 2)
    #     self.assertEqual(len(next(x)), 1)

    def test_combine_dead(self):
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
        p = Pareto(data1).combine_dead(Pareto(data2))
        self.assertEqual(len(p.data), 2)

        d = p.data
        # Column "A" should be 4, 6
        self.assertEqual(d["A"].tolist(), [4, 6])
        # Column "B" should be 6, 4
        self.assertEqual(d["B"].tolist(), [6, 4])
        # Column UTIL should be 3, 3
        self.assertEqual(d[OCCUPANCY].tolist(), [3, 3])

    def test_combine_live(self):
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                OCCUPANCY: [3, 3, 3],
                DATA_SIZE("T1"): [1, 2, 3],
                NUM_ELEMS("T1"): [3, 2, 1],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                OCCUPANCY: [3, 3, 1],
                DATA_SIZE("T1"): [2, 2, 2],
                NUM_ELEMS("T1"): [1, 1, 1],
                MAPPING: [{"B": "B"}] * 3,
            }
        )
        p = Pareto.combine_live_all([Pareto(data1), Pareto(data2)])

        self.assertEqual(len(p.data), 2)

        d = p.data

        # Column "A" should be 4, 6
        self.assertEqual(d["A"].tolist(), [4, 6])
        # Column "B" should be 6, 4
        self.assertEqual(d["B"].tolist(), [6, 4])
        self.assertEqual(d[OCCUPANCY].tolist(), [4 - 3 - 2 + 6, 4 - 4 - 2 + 4])
        self.assertEqual(d[DATA_SIZE("T1")].tolist(), [2, 2])
        self.assertEqual(d[NUM_ELEMS("T1")].tolist(), [3, 2])


if __name__ == "__main__":
    unittest.main()
