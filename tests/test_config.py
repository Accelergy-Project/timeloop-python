import logging
from pathlib import Path
import unittest

import bindings
from bindings.config import Configurator

from pytimeloop.config import Config
from pytimeloop.mapping import Mapping
from pytimeloop.model import ArchSpecs, SparseOptimizationInfo
from pytimeloop.problem import Workload

from .util import load_configs, TEST_TMP_DIR, gather_yaml_configs


class ConfigTest(unittest.TestCase):
    def test_config(self):
        CONFIG_DIR = Path('01-model-conv1d-2level')
        PATHS = ['arch/*.yaml',
                 'map/conv1d-2level-os.map.yaml',
                 'prob/*.yaml']
        TMP_DIR = TEST_TMP_DIR / 'AcceleratorTest'
        SEMI_QUAL_PREFIX = 'timeloop-model'
        yaml_str = gather_yaml_configs(CONFIG_DIR, PATHS)
        configurator = Configurator.from_yaml_str(yaml_str)

        self.arch_specs = configurator.get_arch_specs()
        self.workload = configurator.get_workload()
        self.mapping = configurator.get_mapping()
        self.sparse_opts = configurator.get_sparse_opts()