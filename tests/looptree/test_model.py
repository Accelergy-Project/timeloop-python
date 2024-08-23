import unittest
from pathlib import Path

import islpy as isl

from bindings.config import Config
from bindings.looptree import LooptreeModelApp
from pytimeloop.looptree.des import deserialize_looptree_output
from.make_model_app import make_model_app
from tests.util import TEST_TMP_DIR, gather_yaml_configs

class LooptreeModelAppTest(unittest.TestCase):
    def test_model_with_two_level_mm(self):
        self.check_model_app(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR
        )

    @staticmethod
    def make_model_app(config_dir, paths, tmp_path):
        yaml_str = gather_yaml_configs(config_dir, paths)
        config = Config(yaml_str, 'yaml')
        model = LooptreeModelApp(config, str(tmp_path), 'looptree-model')
        return model
    
    def check_model_app(
        self, config_dir, paths, tmp_path
    ):
        model = self.make_model_app(config_dir, paths, tmp_path)
        result = model.run()

        self.assertEqual(
            {
                0: '{ [i0, i1, i2, i3] -> 3 : i1 = 0 and 0 <= i0 <= 2 and 0 <= i2 <= 3 and 0 <= i3 <= 1 }',
                1: '{ [i0, i1, i2, i3] -> 3 : i1 = 1 and 0 <= i0 <= 2 and 0 <= i2 <= 3 and 0 <= i3 <= 7 }'
            },
            result.ops
        )
        self.assertEqual(
            {
                (0, 0, 0): '{ [i0] -> 18 : i0 = 0 }',
                (0, 1, 0): '{ [i0] -> 8 : i0 = 0 }',
                (0, 3, 1): '{ [i0] -> 32 : i0 = 0 }',
                (0, 4, 1): '{ [i0] -> 72 : i0 = 0 }',
                (1, 0, 0): '{ [i0, i1, i2] -> 6 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
                (1, 1, 0): '{ [i0, i1] -> 8 : i0 = 0 and i1 = 0 }',
                (1, 3, 1): '{ [i0, i1] -> 32 : i0 = 0 and i1 = 0 }',
                (1, 4, 1): '{ [i0, i1, i2] -> 24 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
                (2, 2, 0): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }',
                (2, 2, 1): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }'
            },
            result.fill
        )
        self.assertEqual(
            {
            (0, 0, 0): '{ [i0] -> 18 : i0 = 0 }',
            (0, 1, 0): '{ [i0] -> 8 : i0 = 0 }',
            (0, 3, 1): '{ [i0] -> 32 : i0 = 0 }',
            (0, 4, 1): '{ [i0] -> 72 : i0 = 0 }',
            (1, 0, 0): '{ [i0, i1, i2] -> 6 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
            (1, 1, 0): '{ [i0, i1] -> 8 : i0 = 0 and i1 = 0 }',
            (1, 3, 1): '{ [i0, i1] -> 32 : i0 = 0 and i1 = 0 }',
            (1, 4, 1): '{ [i0, i1, i2] -> 24 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
            (2, 2, 0): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }',
            (2, 2, 1): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }'
            },
            result.occupancy
        )


class TestLooptreeOutputDeserializer(unittest.TestCase):
    def test_deserializer_with_two_level_mm(self):
        self.check_deserializer(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR
        )

    @staticmethod
    def make_model_app(config_dir, paths, tmp_path):
        yaml_str = gather_yaml_configs(config_dir, paths)
        return LooptreeModelApp(Config(yaml_str, 'yaml'),
                                str(tmp_path),
                                'looptree-model')
    
    def check_deserializer(
        self, config_dir, paths, tmp_path
    ):
        model = self.make_model_app(config_dir, paths, tmp_path)
        result = model.run()

        des_result = deserialize_looptree_output(result, isl.DEFAULT_CONTEXT)

        self.assertEqual(
            {
                0: '{ [i0, i1, i2, i3] -> 3 : i1 = 0 and 0 <= i0 <= 2 and 0 <= i2 <= 3 and 0 <= i3 <= 1 }',
                1: '{ [i0, i1, i2, i3] -> 3 : i1 = 1 and 0 <= i0 <= 2 and 0 <= i2 <= 3 and 0 <= i3 <= 7 }'
            },
            {
                k: v.to_str() for k, v in des_result.ops.items()
            }
        )
        self.assertEqual(
            {
                (0, 0, 0): '{ [i0] -> 18 : i0 = 0 }',
                (0, 1, 0): '{ [i0] -> 8 : i0 = 0 }',
                (0, 3, 1): '{ [i0] -> 32 : i0 = 0 }',
                (0, 4, 1): '{ [i0] -> 72 : i0 = 0 }',
                (1, 0, 0): '{ [i0, i1, i2] -> 6 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
                (1, 1, 0): '{ [i0, i1] -> 8 : i0 = 0 and i1 = 0 }',
                (1, 3, 1): '{ [i0, i1] -> 32 : i0 = 0 and i1 = 0 }',
                (1, 4, 1): '{ [i0, i1, i2] -> 24 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
                (2, 2, 0): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }',
                (2, 2, 1): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }'
            },
            {
                k: v.to_str() for k, v in des_result.fill.items()
            }
        )
        self.assertEqual(
            {
            (0, 0, 0): '{ [i0] -> 18 : i0 = 0 }',
            (0, 1, 0): '{ [i0] -> 8 : i0 = 0 }',
            (0, 3, 1): '{ [i0] -> 32 : i0 = 0 }',
            (0, 4, 1): '{ [i0] -> 72 : i0 = 0 }',
            (1, 0, 0): '{ [i0, i1, i2] -> 6 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
            (1, 1, 0): '{ [i0, i1] -> 8 : i0 = 0 and i1 = 0 }',
            (1, 3, 1): '{ [i0, i1] -> 32 : i0 = 0 and i1 = 0 }',
            (1, 4, 1): '{ [i0, i1, i2] -> 24 : i0 = 0 and i1 = 0 and 0 <= i2 <= 2 }',
            (2, 2, 0): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }',
            (2, 2, 1): '{ [i0, i1, i2, i3] -> 12 : i0 = 0 and i1 = 0 and i3 = 0 and 0 <= i2 <= 2 }'
            },
            {
                k: v.to_str() for k, v in des_result.occupancy.items()
            }
        )
