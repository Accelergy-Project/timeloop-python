import unittest

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.looptree.energy import gather_actions
from pytimeloop.fastfusion.fastmodel import compile_mapping
from pytimeloop.looptree.mapping_utilities import get_paths

from tests.load_config_mixin import LoadConfigMixin


class TestLooptreeFastModel(LoadConfigMixin, unittest.TestCase):
    def test_with_fused(self):
        config, spec = self.load_config([
            'looptree-test-fused.yaml',
            'cascaded_mm.workload.yaml',
            'three_level.arch.yaml'
        ])

        BINDINGS = {
            0: 'MainMemory',
            1: 'GlobalBuffer',
            2: 'GlobalBuffer',
            3: 'GlobalBuffer',
            4: 'MACC'
        }

        workload = LooptreeWorkload.parse_cfg(config.root['problem'])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

        for path in get_paths(spec.mapping['nodes']):
            if path[-1]['einsum'] == 'Fc1':
                mapping = path
                break

        result = compile_mapping(mapping, workload, analyzer)
        print(result)
