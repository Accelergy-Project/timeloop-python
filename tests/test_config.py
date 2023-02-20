import logging
from pathlib import Path
import unittest

import bindings
from bindings.config import Configurator

from .util import TEST_TMP_DIR, gather_yaml_configs


class ConfigTest(unittest.TestCase):
    def test_config(self):
        CONFIG_DIR = Path('01-model-conv1d-2level')
        PATHS = ['arch/*.yaml',
                 'map/conv1d-2level-os.map.yaml',
                 'prob/*.yaml']
        yaml_str = gather_yaml_configs(CONFIG_DIR, PATHS)
        configurator = Configurator.from_yaml_str(yaml_str)

        self.arch_specs = configurator.get_arch_specs()
        self.workload = configurator.get_workload()
        self.mapping = configurator.get_mapping()
        self.sparse_opts = configurator.get_sparse_opts()