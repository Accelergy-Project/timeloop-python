import os
import unittest
from pytimeloop.timeloopfe.v4.processors.constraint_attacher import (
    ConstraintAttacherProcessor,
)
from pytimeloop.timeloopfe.common.processor import Processor
from pytimeloop.timeloopfe.common.nodes import Node, ParseError
from pytimeloop.timeloopfe.v4.specification import Specification
from pytimeloop.timeloopfe.v4.arch import Component, Hierarchical, Leaf, Parallel
from pytimeloop.timeloopfe.v4.constraints import (
    ConstraintGroup,
    Temporal,
    Factors,
)
from pytimeloop.timeloopfe.v4.processors import References2CopiesProcessor


class Refs2CopiesTest(unittest.TestCase):
    def get_spec(self, *args, **kwargs) -> Specification:
        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        args = [os.path.join(this_script_dir, a) for a in args]
        return Specification.from_yaml_files(
            os.path.join(this_script_dir, "arch_nest.yaml"), *args, **kwargs
        )

    # def test_references(self):
    #     spec = self.get_spec(
    #         processors=[References2CopiesProcessor], preserve_references=True
    #     )
    #     self.assertEqual(
    #         id(spec.architecture.nodes[5].attributes),
    #         id(spec.architecture.nodes[7].nodes[0].attributes),
    #     )
    #     self.assertEqual(
    #         id(spec.architecture.nodes[6].nodes),
    #         id(spec.architecture.nodes[7].nodes[1].nodes),
    #     )
    #     spec.architecture.nodes[5]["attributes"]["abc"] = "def"
    #     self.assertIn("abc", spec.architecture.nodes[7].nodes[0]["attributes"])
    #     self.assertEqual(
    #         spec.architecture.nodes[7].nodes[0]["attributes"]["abc"], "def"
    #     )

    def test_break_references(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor], preserve_references=True
        )
        spec.process()
        self.assertNotEqual(
            id(spec.architecture.nodes[5]),
            id(spec.architecture.nodes[7].nodes[0]),
        )
        self.assertNotEqual(
            id(spec.architecture.nodes[7].nodes),
            id(spec.architecture.nodes[7].nodes[1]),
        )

    # def test_mutate_references(self):
    #     spec = self.get_spec(
    #         processors=[References2CopiesProcessor], preserve_references=True
    #     )
    #     spec.architecture.nodes[5]["attributes"]["a"] = "abc"
    #     p = Parallel()
    #     spec.architecture.nodes[6].nodes.append(p)

    #     # Mutatee
    #     self.assertEqual(spec.architecture.nodes[5]["attributes"]["a"], "abc")
    #     self.assertEqual(spec.architecture.nodes[6].nodes[-1], p)

    #     # Reference
    #     self.assertEqual(spec.architecture.nodes[7].nodes[0]["attributes"]["a"], "abc")
    #     self.assertEqual(spec.architecture.nodes[7].nodes[1].nodes[-1], p)

    def test_mutate_broken_references(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor], preserve_references=True
        )
        spec.process()
        spec.architecture.nodes[5]["attributes"]["a"] = "abc"
        p = Parallel()
        spec.architecture.nodes[6].nodes.append(p)

        # Mutatee
        self.assertEqual(spec.architecture.nodes[5]["attributes"]["a"], "abc")
        self.assertEqual(spec.architecture.nodes[6].nodes[-1], p)

        # Reference
        self.assertNotIn("a", spec.architecture.nodes[7].nodes[0]["attributes"])
        self.assertNotEqual(spec.architecture.nodes[7].nodes[1].nodes[-1], p)

    def test_merges(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor], preserve_references=True
        )
        a, b, c = tuple(spec.architecture.nodes[x + 8] for x in range(3))

        self.assertEqual(a.name, "merge_A")
        self.assertEqual(b.name, "merge_B")
        self.assertEqual(a.area_scale, 1)
        self.assertEqual(b.area_scale, 2)
        self.assertEqual(a.attributes["from_A"], 1)
        self.assertEqual(b.attributes["from_B"], 2)
        self.assertListEqual(b.attributes["from_B_list"], [])

        self.assertEqual(a["class"], "storage")
        self.assertEqual(b["class"], "storage")
        self.assertEqual(c["class"], "storage")

        # What is there already takes precedence
        self.assertEqual(c.name, "merge_C")

        # Followed by <<
        self.assertEqual(c.area_scale, 1)

        # Recursively merged from B
        for k in ["from_C", "from_B", "from_B_list"]:
            self.assertIn(k, c.attributes)
        for k in ["from_A", "from_A_list"]:
            self.assertNotIn(k, c.attributes)

    # def test_merges_references_preserved_through_casting(self):
    #     spec = self.get_spec(
    #         processors=[References2CopiesProcessor], preserve_references=True
    #     )
    #     a, b, c = tuple(spec.architecture.nodes[x + 8] for x in range(3))
    #     # Make sure constraints were casted
    #     self.assertIsInstance(a.constraints, ConstraintGroup)
    #     self.assertIsInstance(b.constraints, ConstraintGroup)
    #     self.assertIsInstance(c.constraints, ConstraintGroup)
    #     # And that they are propogated by ID
    #     self.assertEqual(id(c.constraints), id(a.constraints))

    def test_casting_break_references(self):
        spec = self.get_spec(
            processors=[References2CopiesProcessor], preserve_references=True
        )
        a, b, c = tuple(spec.architecture.nodes[x + 8] for x in range(3))
        # Make sure constraints were casted
        self.assertIsInstance(a.constraints, ConstraintGroup)
        self.assertIsInstance(b.constraints, ConstraintGroup)
        self.assertIsInstance(c.constraints, ConstraintGroup)
        spec.process()
        # And that they are propogated by ID
        self.assertNotEqual(id(c.constraints), id(a.constraints))
