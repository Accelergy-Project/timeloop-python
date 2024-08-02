from pathlib import Path
import unittest

from pytimeloop.timeloopfe.v4.specification import Specification


class TestErt(unittest.TestCase):
    TEST_DIR = Path(__file__).parent
    ARCH_NEST_PATH = str(TEST_DIR / 'arch_nest.yaml')
    ERT_PATH = str(TEST_DIR / 'ert.yaml')

    def test_ert(self):
        spec = Specification.from_yaml_files(
            TestErt.ARCH_NEST_PATH, TestErt.ERT_PATH
        )

        ert = spec.ERT.tables

        self.assertEqual(ert[0].name, 'ComponentA')
        self.assertEqual(ert[0].actions[0].name, 'ActionA1')
        self.assertEqual(ert[0].actions[0].energy, 1.0)
        self.assertEqual(ert[0].actions[1].name, 'ActionA2')
        self.assertEqual(ert[0].actions[1].energy, 2.0)

        self.assertEqual(ert[1].name, 'ComponentB')
        self.assertEqual(ert[1].actions[0].name, 'ActionB1')
        self.assertEqual(ert[1].actions[0].energy, 3.0)
