import unittest
from pathlib import Path

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from bindings.config import Configurator
from pytimeloop.app import ModelApp
from pytimeloop.accelergy_interface import invoke_accelergy

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class ModelAppTest(unittest.TestCase):
    @staticmethod
    def make_model_app(config_dir, paths, tmp_path):
        yaml_str = gather_yaml_configs(config_dir, paths)
        if "ERT" not in yaml.load(yaml_str, Loader=Loader):
            with open("tmp-accelergy.yaml", "w") as f:
                f.write(yaml_str)
            result = invoke_accelergy(
                ["tmp-accelergy.yaml"],
                "tmp_accelergy_output"
            )
            yaml_str += result.art
            yaml_str += result.ert
        return ModelApp(yaml_str, default_out_dir=str(tmp_path))

    def check_model_app(
            self, config_dir, paths, ref_cycles, ref_area, ref_energy, tmp_path
        ):
        model = self.make_model_app(config_dir, paths, tmp_path)
        eval_result = model.run()

        self.assertEqual(eval_result.cycles, ref_cycles)
        self.assertAlmostEqual(eval_result.area, ref_area, 1)
        self.assertAlmostEqual(eval_result.energy, ref_energy, 1)

    def test_model_with_two_level_mm(self):
        self.check_model_app(
            Path('test_configs'),
            ['*.yaml'],
            64,
            0,
            19590.9,
            TEST_TMP_DIR,
        )
