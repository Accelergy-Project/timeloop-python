import unittest

from pytimeloop.fastfusion.layerdeduplication import is_equivalent
from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from tests.load_config_mixin import LoadConfigMixin


class TestLayerDeduplication(LoadConfigMixin, unittest.TestCase):
    def test_is_equivalent_mismatch(self):
        config, spec = self.load_config([
            'four_level.arch.yaml',
            'cascaded_mm.workload.yaml'
        ])
        workload = LooptreeWorkload.parse_cfg(config.root['problem'])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

        rank_renaming, tensor_renaming = \
            is_equivalent(0, 1, workload, analyzer)

        self.assertIs(rank_renaming, None)
        self.assertIs(tensor_renaming, None)

    def test_is_equivalent_match(self):
        config, spec = self.load_config([
            'four_level.arch.yaml',
            'cascaded_mm_32.workload.yaml'
        ])
        workload = LooptreeWorkload.parse_cfg(config.root['problem'])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

        rank_renaming, tensor_renaming = \
            is_equivalent(0, 1, workload, analyzer)

        self.assertEqual(rank_renaming, {0: 9, 1: 10, 2: 11})
        self.assertEqual(tensor_renaming, {0: 2, 1: 3, 2: 4})