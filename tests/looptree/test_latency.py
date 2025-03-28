from pathlib import Path
import unittest

from pytimeloop.looptree.run import run_looptree

from tests.util import TEST_TMP_DIR


class TestLatency(unittest.TestCase):
    def test_fused_sequential(self):
        BINDINGS = {
            0: 'MainMemory',
            1: 'GlobalBuffer',
            2: 'GlobalBuffer',
            3: 'GlobalBuffer',
            4: 'MACC'
        }

        stats = run_looptree(
            Path(__file__).parent.parent / 'test_configs',
            [
                'looptree-test-fused.yaml',
                'cascaded_mm.workload.yaml',
                'three_level.arch.yaml'
            ],
            TEST_TMP_DIR,
            BINDINGS,
            True
        )

        self.assertEqual(54, stats.latency)
