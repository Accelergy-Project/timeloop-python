import unittest
from pathlib import Path
from pytimeloop.app import ModelApp
from pytimeloop.config import Config

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class ModelAppTest(unittest.TestCase):
    def test_model_with_two_level_mm(self):
        self.check_model_app(
            Path(__file__).parent / 'test_configs',
            ['two_level.arch.yaml', 'mapping.yaml', 'mm.workload.yaml'],
            64,
            19590.9,
            TEST_TMP_DIR,
        )

    @staticmethod
    def make_model_app(config_dir, paths, tmp_path):
        yaml_str = gather_yaml_configs(config_dir, paths)
        return ModelApp(Config(yaml_str, 'yaml'), str(tmp_path), 'timeloop-model')

    def check_model_app(
            self, config_dir, paths, ref_cycles, ref_energy, tmp_path
        ):
        model = self.make_model_app(config_dir, paths, tmp_path)
        eval_result = model.run()

        self.assertEqual(eval_result.cycles, ref_cycles)
        self.assertAlmostEqual(eval_result.energy, ref_energy, 1)
