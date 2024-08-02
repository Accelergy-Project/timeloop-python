import os
import unittest
from pytimeloop.timeloopfe.v4.processors import References2CopiesProcessor
from pytimeloop.timeloopfe.v4.specification import Specification
from pytimeloop.timeloopfe.v4.processors.constraint_macro import (
    ConstraintMacroProcessor,
)
from pytimeloop.timeloopfe.v4 import constraints


class TestConstraintMacroProcessorParsing(unittest.TestCase):
    def get_spec(self, **kwargs) -> Specification:
        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        return Specification.from_yaml_files(
            os.path.join(this_script_dir, "arch_nest.yaml"), **kwargs
        )

    def test_keep_only(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor, ConstraintMacroProcessor]
        )
        ds = constraints.Dataspace(keep_only=["dataspace_A", "dataspace_B"])
        spec.constraints.targets.append(ds)
        spec.process()
        ds = spec.constraints.targets[-1]
        self.assertSetEqual(set(ds.keep), set(["dataspace_A", "dataspace_B"]))
        self.assertSetEqual(set(ds.bypass), set(["dataspace_C"]))

    def test_bypass_only(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor, ConstraintMacroProcessor]
        )
        ds = constraints.Dataspace(bypass_only=["dataspace_A"])
        spec.constraints.targets.append(ds)
        spec.process()
        ds = spec.constraints.targets[-1]
        self.assertSetEqual(set(ds.bypass), set(["dataspace_A"]))
        self.assertSetEqual(set(ds.keep), set(["dataspace_B", "dataspace_C"]))

    def test_factors_only(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor, ConstraintMacroProcessor]
        )
        it = constraints.Temporal(factors_only="A=2")
        spec.constraints.targets.append(it)
        spec.process()
        it = spec.constraints.targets[-1]
        self.assertSetEqual(
            set([(f, d) for f, _, d in it.factors.get_split_factors()]),
            set([("A", 2), ("B", 1), ("C", 1)]),
        )

    def test_no_iteration_over_dataspaces(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor, ConstraintMacroProcessor]
        )
        it = constraints.Temporal(no_iteration_over_dataspaces=["dataspace_A"])
        it2 = constraints.Temporal(no_iteration_over_dataspaces=["dataspace_C"])
        spec.constraints.targets.append(it)
        spec.constraints.targets.append(it2)
        spec.process()
        it = spec.constraints.targets[-2]
        it2 = spec.constraints.targets[-1]
        self.assertSetEqual(set(it.factors), {"A=1"})
        self.assertSetEqual(set(it2.factors), {"C=1"})

    def test_constraint_list_in_star(self):
        pds = constraints.ProblemDataspaceList(["*"])
        assert "dataspace_A" in pds
