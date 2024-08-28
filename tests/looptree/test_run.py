from pathlib import Path
import unittest

from pytimeloop.looptree.run import run_looptree

from tests.util import TEST_TMP_DIR


class TestCompleteRun(unittest.TestCase):
    def test_fused_sequential(self):
        BINDINGS = {
            0: 'MainMemory',
            1: 'GlobalBuffer',
            2: 'GlobalBuffer',
            'compute': 'MACC'
        }

        latency, energy = run_looptree(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR,
            BINDINGS,
            True
        )

        self.assertEqual(24, latency)

        ENERGY_REFS = {
            ('MainMemory', 'read'): 266240,
            ('MainMemory', 'write'): 147456,
            ('GlobalBuffer', 'read'): 103342.36,
            ('GlobalBuffer', 'write'): 35009.79,
            ('MACC', 'compute'): 360
        }

        for k, v in energy.items():
            self.assertAlmostEqual(ENERGY_REFS[k], v, 1)
