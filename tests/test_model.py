import unittest

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
            result = invoke_accelergy(["tmp-accelergy.yaml"], "pytimeloop", ".")
            yaml_str += result.art
            yaml_str += result.ert
        return ModelApp(yaml_str, str(tmp_path))

    def check_model_app(
        self, config_dir, paths, ref_area, ref_energy, ref_cycles, tmp_path
    ):
        model = self.make_model_app(config_dir, paths, tmp_path)
        eval_result = model.run_sandboxed()

        self.assertEqual(eval_result.cycles, ref_cycles)
        self.assertAlmostEqual(eval_result.area, ref_area, 1)
        self.assertAlmostEqual(eval_result.energy, ref_energy, 1)

    def test_conv1d_1level(self):
        self.check_model_app(
            "00-model-conv1d-1level",
            ["arch/*.yaml", "map/*.yaml", "prob/*.yaml"],
            1220.1,
            100.87,
            48,
            TEST_TMP_DIR / "model-conv1d-1level",
        )

    def test_conv1d_2level(self):
        self.check_model_app(
            "01-model-conv1d-2level",
            ["arch/*.yaml", "map/conv1d-2level-os.map.yaml", "prob/*.yaml"],
            748186.1,
            340.7,
            48,
            TEST_TMP_DIR / "model-conv1d-2level",
        )
