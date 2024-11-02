import sys
import unittest

from bindings.looptree import LooptreeWorkload

from pytimeloop.fastfusion.mapper.mapper2 import mapper, MacArrayConstraint

from tests.load_config_mixin import LoadConfigMixin
from tests.util import TEST_TMP_DIR


class TestMapper(LoadConfigMixin, unittest.TestCase):
    def test_mapper(self):
        config, spec = self.load_config([
            'cascaded_mm_large.workload.yaml',
            'four_level.arch.yaml'
        ])

        mac_constraint = MacArrayConstraint(
            64,
            64,
            {
                'Fc1': 'Filter1',
                'Fc2': 'Filter2'
            },
            {
                'Fc1': 'M1',
                'Fc2': 'M2'
            },
            {
                'Fc1': 'C1',
                'Fc2': 'C2'
            }
        )

        result = mapper(config,
                        mac_constraint,
                        spec,
                        explore_pe_uneven=True,
                        tmp_path=TEST_TMP_DIR,
                        verbose_stream=sys.stdout)

