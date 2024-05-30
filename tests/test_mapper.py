import unittest
from pathlib import Path

from pytimeloop.config import Config
from pytimeloop.app import MapperApp

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class MapperAppTest(unittest.TestCase):
    def test_mapper_with_two_level_mm(self):
        self.check_mapper_app(
            Path(__file__).parent / 'test_configs',
            [
                'two_level.arch.yaml',
                'mm.workload.yaml',
                'mapper.yaml'
            ],
            TEST_TMP_DIR,
            64,
            9956.5,
        )

    @staticmethod
    def make_mapper_app(config_dir, paths, tmp_path):
        yaml_str = gather_yaml_configs(config_dir, paths)
        return MapperApp(Config(yaml_str, 'yaml'),
                         str(tmp_path),
                         'timeloop-mapper')

    def check_mapper_app(
        self, config_dir, paths, tmp_path, ref_cycles, ref_energy
    ):
        mapper = self.make_mapper_app(config_dir, paths, tmp_path)
        mapper.run()
        eval_result = mapper.get_global_best()

        self.assertEqual(eval_result.stats.cycles, ref_cycles)
        self.assertAlmostEqual(eval_result.stats.energy, ref_energy, 1)
