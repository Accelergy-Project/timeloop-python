import unittest

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.looptree.energy import gather_actions, compute_energy_from_actions
from pytimeloop.looptree.latency import compute_latency
from pytimeloop.looptree.fastmodel import run_fastmodel

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

        result = run_fastmodel(spec.mapping, 0, workload, analyzer)
        actions = gather_actions(result, spec.mapping, workload, BINDINGS)
