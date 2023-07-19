from pathlib import Path
import unittest

from pytimeloop.config import Configurator
from pytimeloop.engine import Accelerator

from tests.util import gather_yaml_configs


class AcceleratorTest(unittest.TestCase):
    def check_eval_stat(self, eval_stat, ref_area, ref_energy, ref_cycles, ref_id=None):
        self.assertAlmostEqual(eval_stat.energy, ref_energy, 1)
        self.assertEqual(eval_stat.cycles, ref_cycles)
        if ref_id is not None:
            self.assertEqual(eval_stat.id, ref_id)

    def test_accelerator_2level(self):
        CONFIG_DIR = Path("01-model-conv1d-2level")
        PATHS = ["arch/*.yaml", "map/conv1d-2level-os.map.yaml", "prob/*.yaml"]
        yaml_str = gather_yaml_configs(CONFIG_DIR, PATHS)
        configurator = Configurator.from_yaml_str(yaml_str)

        self.arch_specs = configurator.get_arch_specs()
        self.workload = configurator.get_workload()
        self.mapping = configurator.get_mapping()
        self.sparse_opts = configurator.get_sparse_opts()

        acc = Accelerator(self.arch_specs)

        eval_stat = acc.evaluate(self.mapping, self.workload, self.sparse_opts)
        self.check_eval_stat(eval_stat, None, 448.3, 48)

    def test_accelerator_3levelspatial(self):
        CONFIG_DIR = Path("04-model-conv1d+oc-3levelspatial")
        PATHS = [
            "arch/*.yaml",
            "map/conv1d+oc+ic-3levelspatial-cp-ws.map.yaml",
            "prob/*.yaml",
        ]
        yaml_str = gather_yaml_configs(CONFIG_DIR, PATHS)
        configurator = Configurator.from_yaml_str(yaml_str)

        self.arch_specs = configurator.get_arch_specs()
        self.workload = configurator.get_workload()
        self.mapping = configurator.get_mapping()
        self.sparse_opts = configurator.get_sparse_opts()

        acc = Accelerator(self.arch_specs)

        eval_stat = acc.evaluate(self.mapping, self.workload, self.sparse_opts)
        self.check_eval_stat(eval_stat, None, 727165.4, 3072)
