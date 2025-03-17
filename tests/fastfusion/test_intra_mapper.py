import unittest
import logging

from pytimeloop.fastfusion.mapper.mapper_snowcat import mapper

from tests.load_config_mixin import LoadConfigMixin
from tests.util import TEST_TMP_DIR

class TestIntraMapper(LoadConfigMixin, unittest.TestCase):
    def test_with_mm(self):
        logging.basicConfig(filename='tests.fastfusion.test_mapper.log', level=logging.DEBUG)
        config, spec = self.load_config([
            'cascaded_mha.workload.yaml',
            'four_level.arch.yaml'
        ])

        result = mapper(config,
                        explore_glb_uneven=True,
                        spec=spec,
                        tmp_path=TEST_TMP_DIR,
                        four_level=True)