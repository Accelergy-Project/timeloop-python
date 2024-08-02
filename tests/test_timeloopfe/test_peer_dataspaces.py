import logging
import os
import unittest
from pytimeloop.timeloopfe.v4.processors.constraint_attacher import (
    ConstraintAttacherProcessor,
)
from pytimeloop.timeloopfe.v4.processors.dataspace2branch import (
    Dataspace2BranchProcessor,
)
from pytimeloop.timeloopfe.v4.specification import Specification
from pytimeloop.timeloopfe.v4.arch import (
    ArchNodes,
    Hierarchical,
    Leaf,
    Parallel,
    Nothing,
)
from pytimeloop.timeloopfe.v4.constraints import Dataspace, ProblemDataspaceList
from typing import Tuple, Union
from pytimeloop.timeloopfe.v4.processors import References2CopiesProcessor


class PeerDataspaceTest(unittest.TestCase):
    def get_spec(self, **kwargs) -> Specification:
        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        return Specification.from_yaml_files(
            os.path.join(this_script_dir, "arch_nest.yaml"), **kwargs
        )

    def dsname(self, index_or_letter: Union[int, str]) -> str:
        if isinstance(index_or_letter, int):
            return f"dataspace_{chr(ord('A')+index_or_letter)}"
        return f"dataspace_{index_or_letter}"

    def get_prepped_spec(self) -> Tuple[Specification, ArchNodes]:
        spec = self.get_spec(
            processors=[
                References2CopiesProcessor,
                ConstraintAttacherProcessor,
                Dataspace2BranchProcessor,
            ]
        )
        # Fully specify branches first
        for peers in spec.architecture.nodes.get_nodes_of_type(Parallel):
            peers.get_nodes_of_type(Dataspace)[0].keep = [
                "dataspace_A",
                "dataspace_B",
                "dataspace_C",
            ]
        # Clear the peers we're testing
        peers: ArchNodes = spec.architecture.nodes[3].nodes
        for ds in peers.get_nodes_of_type(Dataspace):
            ds.keep = []

        return spec, peers

    def test_unique_keeps(self):
        spec, peers = self.get_prepped_spec()
        for i in range(3):
            peers[i].get_nodes_of_type(Dataspace)[0].keep = [self.dsname(i)]

        spec.process()
        for i, subnode in enumerate(peers):
            bypass = [self.dsname(j) for j in range(3) if j != i]
            for ds_constraint in subnode.get_nodes_of_type(Dataspace):
                # Assert the sets are equal
                logging.info("Checking %s", ds_constraint.get_name())
                self.assertSetEqual(set(ds_constraint.bypass), set(bypass))

    def test_deeply_nested_keeps(self):
        spec, peers = self.get_prepped_spec()
        ds_target = peers[4].nodes[1].get_nodes_of_type(Dataspace)[-1]
        all_ds = [self.dsname(i) for i in range(3)]
        ds_target.keep = all_ds
        spec.process()
        for i in range(5):
            keep_check = []
            ds_check = [self.dsname(i) for i in range(3)] if i != 4 else []

            for ds in peers[i].get_nodes_of_type(Leaf):
                ds = ds.constraints.dataspace
                kc = keep_check if ds is not ds_target else all_ds
                self.assertSetEqual(set(ds.keep), set(kc))
                self.assertSetEqual(set(ds.bypass), set(ds_check))

    def test_empty_unknown_target(self):
        spec, peers = self.get_prepped_spec()
        for i in range(2):
            dataspaces = [self.dsname(k) for k in range(3) if k != i]
            for ds in peers[i].get_nodes_of_type(Dataspace):
                ds.bypass = dataspaces
        peers[2] = Hierarchical()
        with self.assertRaises(ValueError):
            spec.process()

    def test_empty_one_target(self):
        spec, peers = self.get_prepped_spec()
        for i in range(3):
            dataspaces = [self.dsname(k) for k in range(3) if k != i]
            for ds in peers[i].get_nodes_of_type(Dataspace):
                ds.bypass = ProblemDataspaceList(dataspaces)
                ds.keep = ProblemDataspaceList([self.dsname(i)])
        del peers[4]
        del peers[3]
        spec.process()
        for i in range(2):
            keep_check = [self.dsname(i)]
            bypass = [self.dsname(k) for k in range(3) if k != i]
            for ds in peers[i].get_nodes_of_type(Dataspace):
                self.assertSetEqual(set(ds.keep), set(keep_check))
                self.assertSetEqual(set(ds.bypass), set(bypass))

    def test_nested_peers(self):
        spec, peers = self.get_prepped_spec()
        peers[0].constraints.dataspace.keep = [self.dsname(0)]

        sub_peers = peers[3].nodes
        sub_peers[0].constraints.dataspace.keep = [self.dsname(1)]
        sub_peers[1].constraints.dataspace.keep = [self.dsname(2)]
        spec.process()
        for i in range(3):
            keep_check = [self.dsname(i)]
            bypass = [self.dsname(k) for k in range(3) if k != i]
            target = peers[i] if i == 0 else sub_peers[i - 1]
            for ds in target.get_nodes_of_type(Dataspace):
                self.assertSetEqual(set(ds.keep), set(keep_check))
                self.assertSetEqual(set(ds.bypass), set(bypass))
        for i in range(len(peers)):
            if i == 0 or i == 3:
                continue
            bypass = [self.dsname(k) for k in range(3)]
            for ds in peers[i].get_nodes_of_type(Dataspace):
                self.assertSetEqual(set(ds.bypass), set(bypass))
                self.assertSetEqual(set(ds.keep), set([]))

    def test_nested_multi_keep_error(self):
        spec, peers = self.get_prepped_spec()
        peers[0].constraints.dataspace.keep = [self.dsname(0)]
        sub_peers = peers[3].nodes
        sub_peers[0].constraints.dataspace.keep = [
            self.dsname(1),
            self.dsname(2),
        ]
        sub_peers[1].constraints.dataspace.keep = [self.dsname(2)]
        with self.assertRaises(ValueError):
            spec.process()

    def test_nested_ds_missing_error(self):
        spec, peers = self.get_prepped_spec()
        peers[0].constraints.dataspace.keep = [self.dsname(0)]
        sub_peers = peers[3].nodes
        sub_peers[0].constraints.dataspace.keep = [self.dsname(1)]
        with self.assertRaises(ValueError):
            spec.process()

    def test_nothing_error_fix(self):
        spec, peers = self.get_prepped_spec()
        peers[0].constraints.dataspace.keep = [self.dsname(0)]
        sub_peers = peers[3].nodes
        sub_peers[0].constraints.dataspace.keep = [self.dsname(1)]
        sub_peers[1] = Nothing()
        sub_peers[1].constraints.dataspace.keep = [self.dsname(2)]
        spec.process()
        for i in range(3):
            keep_check = [self.dsname(i)]
            bypass = [self.dsname(k) for k in range(3) if k != i]
            target = peers[i] if i == 0 else sub_peers[i - 1]
            for ds in target.get_nodes_of_type(Dataspace):
                self.assertSetEqual(set(ds.keep), set(keep_check))
                self.assertSetEqual(set(ds.bypass), set(bypass))
        for i in range(len(peers)):
            if i == 0 or i == 3:
                continue
            bypass = [self.dsname(k) for k in range(3)]
            for ds in peers[i].get_nodes_of_type(Dataspace):
                self.assertSetEqual(set(ds.bypass), set(bypass))
                self.assertSetEqual(set(ds.keep), set([]))
