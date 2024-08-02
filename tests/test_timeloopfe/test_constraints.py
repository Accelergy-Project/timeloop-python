import unittest
from pytimeloop.timeloopfe.v4 import constraints


class ConstraintTest(unittest.TestCase):
    def test_constraint_combine_dataspaces(self):
        ds = constraints.Dataspace(keep=["1", "2", "3"], bypass=["a", "b", "c"])
        ds2 = constraints.Dataspace(keep=["4", "5", "6"], bypass=["d", "e", "f"])
        ds.combine(ds2)
        self.assertSetEqual(set(ds.keep), set(["1", "2", "3", "4", "5", "6"]))
        self.assertSetEqual(set(ds.bypass), set(["a", "b", "c", "d", "e", "f"]))

    def test_constraint_combine_factors(self):
        temporal = constraints.Temporal(factors="A=1 B=2 C=3")
        temporal2 = constraints.Temporal(factors="D=4 E=5 F=6")
        temporal.combine(temporal2)
        self.assertSetEqual(
            set([(f, d) for f, _, d in temporal.factors.get_split_factors()]),
            set([(l, int(i) + 1) for i, l in enumerate("ABCDEF")]),
        )

    def test_constraint_combine_diffprops(self):
        ds1 = constraints.Dataspace(keep=["1", "2", "3"])
        ds2 = constraints.Dataspace(bypass=["a", "b", "c"])
        ds1.combine(ds2)
        self.assertSetEqual(set(ds1.keep), set(["1", "2", "3"]))
        self.assertSetEqual(set(ds1.bypass), set(["a", "b", "c"]))

    def test_combine_fail_different_types(self):
        with self.assertRaises(ValueError):
            constraints.Dataspace().combine(constraints.Temporal())

    def test_combine_fail_factor_double(self):
        with self.assertRaises(ValueError):
            constraints.Temporal(factors="A=1 B=2").combine(
                constraints.Temporal(factors="A=3 B=4")
            )

    def test_combine_fail_keep_bypass(self):
        with self.assertRaises(ValueError):
            constraints.Dataspace(keep=["A"]).combine(
                constraints.Dataspace(bypass=["A"])
            )


if __name__ == "__main__":
    unittest.main()
