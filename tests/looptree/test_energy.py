import unittest
from pathlib import Path
import io


from ruamel.yaml import YAML
yaml = YAML(typ='safe')
import islpy as isl

from pytimeloop.looptree.des import deserialize_looptree_output
from pytimeloop.looptree.energy import compute_energy_from_actions, gather_actions

from tests.util import TEST_TMP_DIR
from .make_model_app import make_model_app

class LooptreeModelAppTest(unittest.TestCase):
    def test_model_with_two_level_mm(self):
        self.check_energy(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR
        )


    def check_energy(self,
                     config_dir: Path,
                     config_names: list[str],
                     tmp_dir: Path):
        model, spec, workload = make_model_app(config_dir,
                                               config_names,
                                               tmp_dir)

        ert = spec.ERT
        result = deserialize_looptree_output(model.run(), isl.DEFAULT_CONTEXT)

        BINDINGS = {
            0: 'MainMemory',
            1: 'GlobalBuffer',
            2: 'GlobalBuffer',
            'compute': 'MACC'
        }

        actions = gather_actions(result, spec.mapping, workload, BINDINGS)
        energy = compute_energy_from_actions(actions, ert)

        REFS = {
            ('MainMemory', 'read'): 266240,
            ('MainMemory', 'write'): 147456,
            ('GlobalBuffer', 'read'): 103342.36,
            ('GlobalBuffer', 'write'): 35009.79,
            ('MACC', 'compute'): 360
        }

        for k, v in energy.items():
            self.assertAlmostEqual(REFS[k], v, 1)
