import os
import unittest
from pytimeloop.timeloopfe.v4.constraints import Temporal
from pytimeloop.timeloopfe.v4.specification import Specification
from pytimeloop.timeloopfe.v4.arch import Leaf
from pytimeloop.timeloopfe.v4.processors.constraint_attacher import (
    ConstraintAttacherProcessor,
)
from pytimeloop.timeloopfe.v4.processors import References2CopiesProcessor


class TestConstraintAttach(unittest.TestCase):
    def get_spec(self, **kwargs) -> Specification:
        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        return Specification.from_yaml_files(
            os.path.join(this_script_dir, "arch_nest.yaml"),
            **kwargs,
            processors=[
                References2CopiesProcessor,
                ConstraintAttacherProcessor,
            ],
        )

    def test_constraint_attaching(self):
        spec = self.get_spec()
        new_constraint_a = Temporal(target="Peer_Hier_A", factors="A=1")
        new_constraint_b = Temporal(target="Peer_Hier_B", factors="B=1")
        spec.constraints.targets.append(new_constraint_a)
        spec.constraints.targets.append(new_constraint_b)
        spec.process()
        for node in spec.architecture.get_nodes_of_type(Leaf):
            if node.name == "Peer_Hier_A":
                self.assertSetEqual(set(node.constraints.temporal.factors), {"A=1"})
            elif node.name == "Peer_Hier_B":
                self.assertSetEqual(set(node.constraints.temporal.factors), {"B=1"})
            else:
                self.assertEqual(set(node.constraints.temporal.factors), set())

    def test_constraint_attaching_combine(self):
        spec = self.get_spec()
        new_constraint_a = Temporal(target="Peer_Hier_A", factors="A=1")
        new_constraint_b = Temporal(target="Peer_Hier_A", factors="B=1")
        spec.constraints.targets.append(new_constraint_a)
        spec.constraints.targets.append(new_constraint_b)
        spec.process()
        for node in spec.architecture.get_nodes_of_type(Leaf):
            if node.name == "Peer_Hier_A":
                self.assertSetEqual(
                    set(node.constraints.temporal.factors),
                    set(["A=1", "B=1"]),
                )
            else:
                self.assertSetEqual(set(node.constraints.temporal.factors), set())
