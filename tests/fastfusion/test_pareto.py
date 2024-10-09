import unittest

from pytimeloop.fastfusion.pareto import *


class ParetoTest(unittest.TestCase):
    def test_pareto(self):
        od1 = OpData(fzs(["A"]))
        data = pd.DataFrame({"A": [1, 2], OCCUPANCY: [2, 1], MAPPING: [{"A": "A"}] * 2})
        Pareto({od1: data})

    def test_vertical_combine(self):
        od1 = OpData(fzs(["A"]))
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                OCCUPANCY: [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od2 = OpData(fzs(["A"]))
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                OCCUPANCY: [3, 3, 1],
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
                OCCUPANCY: [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od2 = OpData(fzs(["B"]))
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                OCCUPANCY: [3, 3, 1],
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
                OCCUPANCY: [3, 3, 3],
                MAPPING: [{"A": "A"}] * 3,
            }
        )
        od2 = OpData(fzs(["B"]))
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                OCCUPANCY: [3, 3, 1],
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
        self.assertEqual(d[OCCUPANCY].tolist(), [3, 3])

    def test_combine_live(self):
        od1 = OpData(fzs(["A"]), fzs(["T1"]))
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
        od2 = OpData(fzs(["B"]), fzs(["T1"]))
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
        p = Pareto({od1: data1, od2: data2})
        p._combine_by_partition(fzs(["A", "B"]))

        self.assertEqual(len(p.data), 1)
        self.assertEqual(len(next(iter(p.data.values()))), 2)

        d = next(iter(p.data.values()))

        # Column "A" should be 4, 6
        self.assertEqual(d["A"].tolist(), [4, 6])
        # Column "B" should be 6, 4
        self.assertEqual(d["B"].tolist(), [6, 4])
        self.assertEqual(d[OCCUPANCY].tolist(), [4 - 3 - 2 + 6, 4 - 4 - 2 + 4])
        self.assertEqual(d[DATA_SIZE("T1")].tolist(), [2, 2])
        self.assertEqual(d[NUM_ELEMS("T1")].tolist(), [3, 2])


if __name__ == "__main__":
    unittest.main()