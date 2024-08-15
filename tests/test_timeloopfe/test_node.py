import os
import unittest
from pathlib import Path

from pytimeloop.timeloopfe.v4.processors.constraint_attacher import (
    ConstraintAttacherProcessor,
)
from pytimeloop.timeloopfe.common.processor import Processor
from pytimeloop.timeloopfe.common.nodes import DictNode, ParseError
from pytimeloop.timeloopfe.v4.specification import Specification
from pytimeloop.timeloopfe.v4.arch import (
    Component,
    Hierarchical,
    Leaf,
    Storage,
    StorageAttributes,
)
from pytimeloop.timeloopfe.v4.constraints import Temporal
from pytimeloop.timeloopfe.v4.processors import (
    Dataspace2BranchProcessor,
    References2CopiesProcessor,
)

from pytimeloop.timeloopfe.common import ParseError, ProcessorError

import pytimeloop.timeloopfe.v4 as tl


class NodeTest(unittest.TestCase):
    def get_spec(self, *args, **kwargs) -> Specification:
        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        args = [os.path.join(this_script_dir, a) for a in args]
        return Specification.from_yaml_files(
            os.path.join(this_script_dir, "arch_nest.yaml"), *args, **kwargs
        )

    def test_from_yaml_path_vs_str(self):
        arch_nest_path = Path(__file__).parent / 'arch_nest.yaml'
        spec = Specification.from_yaml_files(arch_nest_path)
        arch_nest_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'arch_nest.yaml'
        )
        spec2 = Specification.from_yaml_files(arch_nest_path)
        self.assertEqual(spec, spec2)

    def test_missing_key(self):
        with self.assertRaises((KeyError)):
            Component({"name": "test"})

    def test_extra_key(self):
        with self.assertRaises(ParseError):
            Component(
                {"name": "test", "class": "storage", "extra": "abc"}
            ).check_unrecognized()

    def test_unrecognized_tag(self):
        class Tagged:
            pass

        with self.assertRaises((KeyError, ValueError, ParseError)):
            Hierarchical(nodes=[Tagged()]).check_unrecognized()

    def test_unrecognized_type_tag(self):
        class Tagged:
            pass

        x = Tagged()
        x.tag = "!Component"
        with self.assertRaises(TypeError):
            y = Hierarchical()
            y.nodes.append(x)
            y.check_unrecognized()

    def test_unrecognized_type_key(self):
        with self.assertRaises(((TypeError, ParseError))):
            Component({"name": 32, "class": "storage"}).check_unrecognized()

    def test_should_have_been_removed_by(self):
        class Test(Processor):
            def declare_attrs(self):
                super().add_attr(Component, "for_testing_ignoreme", str, "")

            def process(self, spec: Specification):
                for e in spec.get_nodes_of_type(Component):
                    print(f"Checking {e.name} {e.__class__.__name__}")
                    e.pop("for_testing_ignoreme", None)


        spec = self.get_spec(processors=[Test])
        spec.architecture.nodes.insert(
            0, Component({"name": ".", "class": "storage", "for_testing_ignoreme": "."})
        )
        with self.assertRaises(ProcessorError):
            spec.check_unrecognized()
        spec.process()
        spec._required_processors.remove(Dataspace2BranchProcessor)
        spec.process(spec._required_processors)
        spec.check_unrecognized()

    def test_should_have_been_removed_by_multi_spec(self):
        class Test(Processor):
            def declare_attrs(self):
                super().add_attr(Component, "for_testing_ignoreme", str, "")

            def process(self, spec: Specification):
                for e in spec.get_nodes_of_type(Component):
                    print(f"Checking {e.name} {e.__class__.__name__}")
                    e.pop("for_testing_ignoreme", None)

        def setup():
            spec = self.get_spec(processors=[Test])
            spec2 = self.get_spec()
            spec.architecture.nodes.insert(
                0,
                Component(
                    {"name": ".", "class": "storage", "for_testing_ignoreme": "."}
                ),
            )
            spec._required_processors.remove(Dataspace2BranchProcessor)
            spec2._required_processors.remove(Dataspace2BranchProcessor)
            spec.process(spec._required_processors)
            spec2.process(spec2._required_processors)
            return spec, spec2

        spec1, spec2 = setup()
        spec2.check_unrecognized()
        with self.assertRaises(ProcessorError):
            spec1.check_unrecognized()
        spec1.process()
        spec1.check_unrecognized()

        spec1, spec2 = setup()
        with self.assertRaises(ProcessorError):
            spec1.check_unrecognized()
        spec1.process()
        spec2.check_unrecognized()

        spec1, spec2 = setup()
        with self.assertRaises(ProcessorError):
            spec1.check_unrecognized()
        spec1.process()
        spec2.check_unrecognized()

    def test_repeated_key_error(self):
        with self.assertRaises(ParseError):
            spec = self.get_spec("repeated_key_error.yaml")

    def test_multi_list_constraints(self):
        spec = self.get_spec(
            "multi_list_constraints.yaml",
            processors=[
                References2CopiesProcessor,
                ConstraintAttacherProcessor,
            ],
        )
        spec.process()
        for node in spec.architecture.get_nodes_of_type(Leaf):
            if node.name == "Hier_A":
                self.assertSetEqual(
                    set(node.constraints.dataspace.bypass),
                    set(["dataspace_B", "dataspace_C"]),
                )
                self.assertSetEqual(
                    set(node.constraints.dataspace.keep), set(["dataspace_A"])
                )
            elif node.name == "Hier_B":
                self.assertSetEqual(
                    set(node.constraints.dataspace.bypass),
                    set(["dataspace_A", "dataspace_C"]),
                )
                self.assertSetEqual(
                    set(node.constraints.dataspace.keep), set(["dataspace_B"])
                )
            else:
                self.assertSetEqual(set(node.constraints.dataspace.bypass), set())
                self.assertSetEqual(set(node.constraints.dataspace.keep), set())

    def test_repeated_node_init(self):
        t1 = Temporal(factors="A=1 B=2 C=3")
        t2 = Temporal(factors=t1.factors)
        self.assertEqual(t1, t2)

    def test_dash_alias_errors(self):
        x = Temporal(no_reuse=[])
        with self.assertRaises(KeyError):
            x["no-reuse"] = []
        with self.assertRaises(KeyError):
            x.get("no-reuse", None)
        with self.assertRaises(KeyError):
            x.pop("no-reuse", None)
        with self.assertRaises(KeyError):
            x.setdefault("no-reuse", [])

    def test_multi_require_exactly_one(self):
        x = StorageAttributes(datawidth=5, depth=5, width=5, block_size=5, technology=5)
        x.check_unrecognized()
        with self.assertRaises(KeyError):
            x.cluster_size = 5
            x.check_unrecognized()

    def test_multi_require_all_or_none_of(self):
        x = StorageAttributes(datawidth=5, depth=5, width=5, block_size=5, technology=5)
        x.check_unrecognized()
        x.metadata_block_size = 1
        x.metadata_datawidth = 1
        x.metadata_storage_depth = 1
        x.metadata_storage_width = 1
        x.check_unrecognized()
        with self.assertRaises(KeyError):
            del x.metadata_block_size
            del x["metadata_block_size"]
            x.check_unrecognized()

    def get_property_table(self):
        tl.doc.get_property_table(Specification)
        tl.doc.get_property_table(Component)

    def get_property_tree(self):
        tl.doc.get_property_table(Specification)
        tl.doc.get_property_table(Component)
