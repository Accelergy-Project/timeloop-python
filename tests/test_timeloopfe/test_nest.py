import logging
import os
import unittest
from pytimeloop.timeloopfe.v4.specification import Specification
from pytimeloop.timeloopfe.v4.arch import (
    Component,
    Hierarchical,
    Parallel,
    Pipelined,
    Leaf,
)

from pytimeloop.timeloopfe.common import ParseError


class ArchNestTest(unittest.TestCase):
    def get_spec(self, **kwargs) -> Specification:
        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        return Specification.from_yaml_files(
            os.path.join(this_script_dir, "arch_nest.yaml"), **kwargs, processors=[]
        )

    def test_arch_structure(self):
        arch = self.get_spec().architecture.nodes

        def check(location, nest):
            for i, check in enumerate("AB"):
                tocheck = "_".join([n[1:5] for n in nest] + [check])
                logging.info("Checking %s == %s", location[i].name, tocheck)
                self.assertEqual(location[i].name, tocheck)

        check(arch, [])
        for listnode in arch:
            if not isinstance(listnode, list):
                continue
            check(listnode, [listnode.get_tag()])
            for sublist in listnode:
                if not isinstance(sublist, list):
                    continue
                check(sublist, [listnode.get_tag(), sublist.get_tag()])

    def test_types(self):
        arch = self.get_spec().architecture.nodes
        self.assertIsInstance(arch[0], Component)
        self.assertIsInstance(arch[1], Component)
        self.assertIsInstance(arch[2], Hierarchical)
        self.assertIsInstance(arch[3], Parallel)
        self.assertIsInstance(arch[4], Pipelined)
        for i in range(2, 5):
            self.assertIsInstance(arch[i].nodes[0], Component)
            self.assertIsInstance(arch[i].nodes[1], Component)
            self.assertIsInstance(arch[i].nodes[2], Hierarchical)
            self.assertIsInstance(arch[i].nodes[3], Parallel)
            self.assertIsInstance(arch[i].nodes[4], Pipelined)
            for j in range(2, 5):
                self.assertIsInstance(arch[i].nodes[j].nodes[0], Component)
                self.assertIsInstance(arch[i].nodes[j].nodes[1], Component)

    def test_mutate_arch_attrs(self):
        arch = self.get_spec().architecture.nodes
        for l in arch.get_nodes_of_type(Leaf):
            l.attributes["test"] = "test"

        def assert_test_attr(node):
            if isinstance(node, Leaf):
                logging.info("Checking %s attributes", node.name)
                self.assertEqual(node.attributes["test"], "test")

        arch.recursive_apply(assert_test_attr)
        arch.check_unrecognized(ignore_should_have_been_removed_by=1)

    def test_mutate_arch(self):
        arch = self.get_spec().architecture.nodes
        mutators = arch.get_setters_for_keytag("!Component")
        for elem, setter in mutators:
            if "B" in elem.name:
                setter(None)

        def assert_test_attr(node):
            if isinstance(node, Component):
                logging.info("Checking %s name", node.name)
                self.assertNotIn("B", node.name)

        arch.recursive_apply(assert_test_attr)
        arch.check_unrecognized(ignore_empty=True)

    def test_mutate_arch_fail_recognized_unrecognized_tag(self):
        arch = self.get_spec().architecture.nodes

        class UnrecognizedTag:
            pass

        arch[0] = UnrecognizedTag()
        with self.assertRaises((ParseError)):
            arch.check_unrecognized()
        arch[0].tag = "!Component"
        with self.assertRaises((TypeError)):
            arch.check_unrecognized()
        arch[0] = Component({"name": "test", "class": "storage"})
        arch.check_unrecognized(ignore_should_have_been_removed_by=1)

    def test_mutate_arch_fail_recognized_unrecognized_key(self):
        arch = self.get_spec().architecture.nodes
        arch[0]["unrecognized"] = "123"
        with self.assertRaises((ParseError)):
            arch.check_unrecognized()
        with self.assertRaises((ParseError)):
            arch.check_unrecognized(ignore_empty=True)
        arch[0]["unrecognized"] = None
        arch.check_unrecognized(ignore_empty=True)
        with self.assertRaises((ParseError)):
            arch.check_unrecognized()

    def test_missing_attribute(self):
        with self.assertRaises((KeyError)):
            elem = Component({"name": "test"})
