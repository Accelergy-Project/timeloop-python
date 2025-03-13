import unittest

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.fastfusion.mapper.per_einsum_subspaces.subspaces.loops import make_spatial_fors, make_temporal_fors
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.subspaces.linear_mapping import LinearMapping
from pytimeloop.fastfusion.mapper.constraints import DataflowConstraint, PerEinsumDataflowConstraint

from tests.load_config_mixin import LoadConfigMixin


class TestDataflowConstraint(LoadConfigMixin, unittest.TestCase):
    def setUp(self):
        config, spec = self.load_config([
            'cascaded_mm_multi_32.workload.yaml',
            'four_level.arch.yaml'
        ])
        self.workload = LooptreeWorkload.parse_cfg(config.root['problem'])

    def test_parser(self):
        pattern = {
            'Fc1': ['P1', 'M1', 'C1'],
            'Fc2': ['*', 'M2'],
            'Fc3': ['*', 'M3', '*'],
            'Fc4': ['P4', '/', '*']
        }
        constraint = DataflowConstraint.parse(pattern)
        self.assertEqual(
            constraint.einsum_to_constraint['Fc1'].disallowed_ranks,
            set()
        )
        self.assertEqual(
            constraint.einsum_to_constraint['Fc2'].rank_order,
            ['*', 'M2']
        )
        self.assertEqual(
            constraint.einsum_to_constraint['Fc4'].disallowed_ranks,
            {'P4'}
        )


class TestTemporalLoop(LoadConfigMixin, unittest.TestCase):
    def setUp(self):
        config, spec = self.load_config([
            'cascaded_mm_multi_32.workload.yaml',
            'four_level.arch.yaml'
        ])
        self.workload = LooptreeWorkload.parse_cfg(config.root['problem'])

    def test_without_constraint(self):
        original = LinearMapping()
        count = 0
        for mapping in make_temporal_fors(original, ranks=['P1', 'M1', 'C1']):
            count += 1
            for node in mapping:
                self.assertTrue(node['rank'] in ['P1', 'M1', 'C1'])
        self.assertEqual(count, 16)

    def test_with_constraint(self):
        original = LinearMapping()
        constraint = PerEinsumDataflowConstraint(disallowed_ranks={'P1'},
                                                 rank_order=['*'])
        count = 0
        for mapping in make_temporal_fors(original,
                                          ranks=['P1', 'M1', 'C1'],
                                          dataflow_constraint=constraint):
            count += 1
            for node in mapping:
                self.assertTrue(node['rank'] in ['M1', 'C1'])
        self.assertEqual(count, 5)