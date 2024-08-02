from pathlib import Path
import unittest

from pytimeloop.timeloopfe.v4.specification import Specification


class TestArt(unittest.TestCase):
    TEST_DIR = Path(__file__).parent
    ARCH_NEST_PATH = str(TEST_DIR / 'arch_nest.yaml')
    ART_PATH = str(TEST_DIR / 'art.yaml')

    def test_art(self):
        spec = Specification.from_yaml_files(
            TestArt.ARCH_NEST_PATH, TestArt.ART_PATH
        )
        self.assertEqual(spec.ART.tables[0].name, "ComponentA")
        self.assertEqual(spec.ART.tables[0].area, 1.0)
        self.assertEqual(spec.ART.tables[1].name, "ComponentB")
        self.assertEqual(spec.ART.tables[1].area, 2.0)