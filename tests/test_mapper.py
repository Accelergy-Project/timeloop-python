import unittest

from pytimeloop.app import MapperApp

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class MapperAppTest(unittest.TestCase):
    @staticmethod
    def make_mapper_app(config_dir, paths, tmp_path):
        config = gather_yaml_configs(config_dir, paths)
        return MapperApp(config, str(tmp_path))

    def check_mapper_app(
        self, config_dir, paths, tmp_path, ref_area, ref_energy, ref_cycles
    ):
        mapper = self.make_mapper_app(config_dir, paths, tmp_path)
        eval_result, mapping = mapper.run()

        self.assertEqual(eval_result.cycles, 1536)

    def test_conv1d_3level(self):
        self.check_mapper_app(
            "05-mapper-conv1d+oc-3level",
            [
                "arch/3level.arch.yaml",
                "mapper/exhaustive.mapper.yaml",
                "constraints/conv1d+oc-3level-freebypass.constraints.yaml",
                "prob/conv1d+oc.prob.yaml",
            ],
            TEST_TMP_DIR / "mapper-conv1d-3level",
            0,
            0,
            0,
        )
