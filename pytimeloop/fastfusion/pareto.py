import unittest
from paretoset import paretoset
import pandas as pd
from dataclasses import dataclass
from util import fzs

MAPPING = "__Mappings"

UTIL = lambda x="": f"__{x} Utilization"
DATA_SIZE = lambda x: f"__{x} Data Size"
NUM_ELEMS = lambda x: f"__{x} Num Elems"

MERGE_SUFFIXES = ["_RIGHT_MERGE", "_LEFT_MERGE"]


def suffix_access(d, c):
    return d[[c + MERGE_SUFFIXES[0], c + MERGE_SUFFIXES[1]]]


def sum_columns(d, columns):
    for c in columns:
        if not c.startswith("__") or c == UTIL():
            d[c] = suffix_access(d, c).sum(axis=1)


def merge_cross(
    a1: pd.DataFrame,
    a2: pd.DataFrame,
    util_combine_sum: bool,
    tensors: set[str] = fzs(),
) -> pd.DataFrame:
    d = pd.merge(a1, a2, how="cross", suffixes=MERGE_SUFFIXES)
    sum_columns(d, a2.columns)

    u = suffix_access(d, UTIL())
    d[UTIL()] = u.sum(axis=1) if util_combine_sum else u.max(axis=1)

    for t in tensors:
        min_data_size = suffix_access(d, DATA_SIZE(t)).min(axis=1)
        min_num_elems = suffix_access(d, NUM_ELEMS(t)).min(axis=1)
        max_data_size = suffix_access(d, DATA_SIZE(t)).max(axis=1)
        max_num_elems = suffix_access(d, NUM_ELEMS(t)).max(axis=1)
        d[DATA_SIZE(t)], d[NUM_ELEMS(t)] = max_data_size, max_num_elems
        d[UTIL()] -= min_data_size * min_num_elems

    d = Pareto.pareto(d)
    s = suffix_access(d, MAPPING)
    c0, c1 = s.columns
    d[MAPPING] = s.apply(lambda row: {**row[c0], **row[c1]}, axis=1)
    d = d[[c for c in d.columns if not any(c.endswith(s) for s in MERGE_SUFFIXES)]]

    return d


@dataclass(frozen=True)
class OpData:
    op_names: fzs[str] = fzs()
    tensors: fzs[str] = fzs()

    def __bool__(self) -> bool:
        return bool(self.op_names) or bool(self.tensors)

    def __and__(self, other: "OpData") -> "OpData":
        return OpData(self.op_names & other.op_names, self.tensors & other.tensors)

    def __sub__(self, other: "OpData") -> "OpData":
        return OpData(self.op_names - other.op_names, self.tensors - other.tensors)

    def __or__(self, other: "OpData") -> "OpData":
        return OpData(self.op_names | other.op_names, self.tensors | other.tensors)


class Pareto:
    def __init__(self, data: dict[OpData, pd.DataFrame]):
        self.data: dict[OpData, pd.DataFrame] = data
        self.data = {k: Pareto.pareto(v) for k, v in self.data.items()}

    @staticmethod
    def pareto(data: pd.DataFrame) -> pd.DataFrame:
        d = data[[c for c in data.columns if c == UTIL() or not c.startswith("__")]]
        return data[paretoset(d)].reset_index(drop=True)

    @staticmethod
    def vertical_combine(paretos: list["Pareto"]) -> "Pareto":
        allkeys = set([tuple(sorted(p.data.keys())) for p in paretos])
        if len(allkeys) > 1:
            raise ValueError("Cannot vertical combine pareto sets with different keys")
        newdata = {}
        for k in next(iter(allkeys)):
            newdata[k] = Pareto.pareto(pd.concat([p.data[k] for p in paretos]))
        return Pareto(newdata)

    def combine(self, other: "Pareto") -> "Pareto":
        for k1 in self.data.keys():
            for k2 in other.data.keys():
                if k1 & k2:
                    raise ValueError("Cannot combine pareto sets with overlapping keys")
        return Pareto({**self.data, **other.data})

    @staticmethod
    def get_dead_key():
        return OpData(fzs(), fzs())

    @staticmethod
    def _merge(a, b):
        return pd.merge(a, b, how="cross")

    def _combine_dead(self, key: OpData):
        dead_key = Pareto.get_dead_key()

        if dead_key not in self.data:
            d = self.data.pop(key)
        else:
            d0, d1 = self.data.pop(key), self.data[dead_key]
            d = merge_cross(d0, d1, False)

        cols = [c for c in d.columns if not c.startswith("__")]
        cols += [MAPPING, UTIL()]
        self.data[dead_key] = d[cols].copy()

    def _combine_live(self, key1: OpData, key2: OpData) -> tuple[OpData, pd.DataFrame]:
        d0, tensors0 = self.data.pop(key1), key1.tensors
        d1, tensors1 = self.data.pop(key2), key2.tensors
        d = merge_cross(d0, d1, True, tensors0 & tensors1)
        op_names = key1.op_names | key2.op_names
        tensors = tensors0 | tensors1
        return OpData(fzs(op_names), fzs(tensors)), d

    def _combine_by_partition(self, op_names: set[str]):
        to_combine = [k for k in self.data if k.op_names & op_names]
        for k in to_combine:
            if k.op_names - op_names:
                raise ValueError(f"Partitioning mismatch {k} {op_names}")

        if not to_combine:
            return

        key = to_combine.pop(0)
        while to_combine:
            key2 = to_combine.pop(0)
            key, d = self._combine_live(key, key2)
            self.data[key] = d

    def drop_dead(
        self, live_partitions: list[set[str]], dead_partitions: list[set[str]]
    ):
        [self._combine_by_partition(t) for t in live_partitions + dead_partitions]
        for k in list(self.data.keys()):
            if not any(k.op_names & p for p in live_partitions):
                self._combine_dead(k)
        return self


class ParetoTest(unittest.TestCase):
    def test_pareto(self):
        od1 = OpData(fzs(["A"]))
        data = pd.DataFrame({"A": [1, 2], UTIL(): [2, 1], MAPPING: [{"A": "A"}] * 2})
        Pareto({od1: data})

    def test_vertical_combine(self):
        od1 = OpData(fzs(["A"]))
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                UTIL(): [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od2 = OpData(fzs(["A"]))
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                UTIL(): [3, 3, 1],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od3 = OpData(fzs(["B"]))

        p1 = Pareto({od1: data1})
        self.assertEqual(len(next(iter(p1.data.values()))), 2)
        p2 = Pareto({od2: data2})
        self.assertEqual(len(next(iter(p2.data.values()))), 1)
        p3 = Pareto({od3: data2})

        pd12 = Pareto.vertical_combine([p1, p2])
        self.assertEqual(len(next(iter(pd12.data.values()))), 3)

        with self.assertRaises(ValueError):
            Pareto.vertical_combine([p1, p3])

    def test_combine(self):
        od1 = OpData(fzs(["A"]))
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                UTIL(): [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od2 = OpData(fzs(["B"]))
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                UTIL(): [3, 3, 1],
                MAPPING: [{"B": "B"}] * 3,
            }
        )

        p1 = Pareto({od1: data1})
        p2 = Pareto({od2: data2})

        pd12 = p1.combine(p2)
        x = iter(pd12.data.values())
        self.assertEqual(len(next(x)), 2)
        self.assertEqual(len(next(x)), 1)

    def test_combine_dead(self):
        od1 = OpData(fzs(["A"]))
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                UTIL(): [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od2 = OpData(fzs(["B"]))
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                UTIL(): [3, 3, 1],
                MAPPING: [{"B": "B"}] * 3,
            }
        )
        p = Pareto({od1: data1, od2: data2})
        p._combine_dead(od1)
        p._combine_dead(od2)
        self.assertEqual(len(p.data), 1)
        self.assertEqual(len(next(iter(p.data.values()))), 2)

        d = p.data[Pareto.get_dead_key()]
        # Column "A" should be 4, 6
        self.assertEqual(d["A"].tolist(), [4, 6])
        # Column "B" should be 6, 4
        self.assertEqual(d["B"].tolist(), [6, 4])
        # Column UTIL should be 3, 3
        self.assertEqual(d[UTIL()].tolist(), [3, 3])

    def test_combine_live(self):
        od1 = OpData(fzs(["A"]), fzs(["T1"]))
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                UTIL(): [3, 3, 3],
                DATA_SIZE("T1"): [0.1, 0.2, 0.3],
                NUM_ELEMS("T1"): [0.3, 0.2, 0.1],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od2 = OpData(fzs(["B"]), fzs(["T1"]))
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                UTIL(): [3, 3, 1],
                DATA_SIZE("T1"): [0.2, 0.2, 0.2],
                NUM_ELEMS("T1"): [0.1, 0.1, 0.1],
                MAPPING: [{"B": "B"}] * 3,
            }
        )
        p = Pareto({od1: data1, od2: data2})
        p._combine_by_partition(fzs(["A", "B"]))

        self.assertEqual(len(p.data), 1)
        self.assertEqual(len(next(iter(p.data.values()))), 2)

        d = next(iter(p.data.values()))

        # Column "A" should be 4, 6
        self.assertEqual(d["A"].tolist(), [4, 6])
        # Column "B" should be 6, 4
        self.assertEqual(d["B"].tolist(), [6, 4])
        self.assertEqual(d[UTIL()].tolist(), [4 - 0.1 * 0.1, 4 - 0.2 * 0.1])
        self.assertEqual(d[DATA_SIZE("T1")].tolist(), [0.2, 0.2])
        self.assertEqual(d[NUM_ELEMS("T1")].tolist(), [0.3, 0.2])


if __name__ == "__main__":
    unittest.main()
