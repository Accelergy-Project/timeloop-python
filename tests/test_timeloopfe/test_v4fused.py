from pathlib import Path
import unittest

from pytimeloop.timeloopfe.v4fused.specification import Specification
from pytimeloop.timeloopfe.common.version_transpilers.v4_to_v3 import transpile


class TestV4Fused(unittest.TestCase):
    TEST_DIR = Path(__file__).parent.parent
    CONFIG_PATH = TEST_DIR / 'test_configs'

    def test_parse_and_transpile(self):
        spec = Specification.from_yaml_files([
            TestV4Fused.CONFIG_PATH / 'looptree-test-fused.yaml',
            TestV4Fused.CONFIG_PATH / 'three_level.arch.yaml',
            TestV4Fused.CONFIG_PATH / 'cascaded_mm.workload.yaml'
        ])
        transpiled_spec = transpile(spec)
        self.assertTrue(transpiled_spec['dumped_by_timeloop_front_end'])
        self.assertEqual(
            'MainMemory[1..1]',
            transpiled_spec['architecture']['subtree'][0]['local'][0]['name']
        )
