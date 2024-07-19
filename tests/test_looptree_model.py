import unittest
from pathlib import Path

from bindings.config import Config
from bindings.looptree import LooptreeModelApp

from tests.util import TEST_TMP_DIR, gather_yaml_configs

class LooptreeModelAppTest(unittest.TestCase):
    def test_model_with_two_level_mm(self):
        self.check_model_app(
            Path(__file__).parent / 'test_configs',
            ['looptree-test.yaml'],
            TEST_TMP_DIR
        )

    @staticmethod
    def make_model_app(config_dir, paths, tmp_path):
        yaml_str = gather_yaml_configs(config_dir, paths)
        return LooptreeModelApp(Config(yaml_str, 'yaml'),
                                str(tmp_path),
                                'looptree-model')
    
    def check_model_app(
        self, config_dir, paths, tmp_path
    ):
        model = self.make_model_app(config_dir, paths, tmp_path)
        result = model.run()
        self.assertEqual({0: 72, 1: 288}, result.ops)
        self.assertEqual(
            {
                (0, 0, 0): 18,
                (0, 1, 0): 8,
                (0, 3, 1): 32,
                (0, 4, 1): 72,
                (1, 0, 0): 18,
                (1, 1, 0): 8,
                (1, 3, 1): 32,
                (1, 4, 1): 72,
                (2, 2, 0): 36,
                (2, 2, 1): 36
            },
            result.fill
        )
        self.assertEqual(
            {
                (0, 0, 0): 18,
                (0, 1, 0): 8,
                (0, 3, 1): 32,
                (0, 4, 1): 72,
                (1, 0, 0): 6,
                (1, 1, 0): 8,
                (1, 3, 1): 32,
                (1, 4, 1): 24,
                (2, 2, 0): 12,
                (2, 2, 1): 12
            },
            result.occupancy
        )
