import unittest
from pathlib import Path
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from bindings.config import Config
from bindings.looptree import LooptreeModelApp, LooptreeWorkload
from pytimeloop.isl.top import Context
from pytimeloop.looptree.des import deserialize_looptree_output
from pytimeloop.looptree.accesses import reads_and_writes_from_fill, get_total_accesses

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class TestLoopTreeAccess(unittest.TestCase):
    def test_accesses_with_two_level_mm_fused(self):
        self.check_accesses(
            Path(__file__).parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR,
            {
                (0, 'Fmap1'): 18,
                (0, 'Filter1'): 8,
                (0, 'Filter2'): 32,
                (0, 'Fmap3'): 72
            },
            {
                (0, 'Fmap3'): 72
            }
        )

    def test_accesses_with_two_level_mm_fused(self):
        self.check_accesses(
            Path(__file__).parent / 'test_configs',
            ['looptree-test-unfused.yaml'],
            TEST_TMP_DIR,
            {
                (0, 'Fmap1'): 18,
                (0, 'Filter1'): 8,
                (0, 'Filter2'): 32,
                (0, 'Fmap2'): 36,
                (0, 'Fmap3'): 72
            },
            {
                (0, 'Fmap2'): 36,
                (0, 'Fmap3'): 72
            }
        )

    @staticmethod
    def make_model_app(config_dir, paths, tmp_path):
        yaml_str = gather_yaml_configs(config_dir, paths)
        config = Config(yaml_str, 'yaml')

        return (
            LooptreeModelApp(config,
                             str(tmp_path),
                             'looptree-model'),
            yaml.load(yaml_str),
            LooptreeWorkload.parse_cfg(config.root['problem'])
        )
    
    def check_accesses(
        self, config_dir, paths, tmp_path, read_refs, write_refs
    ):
        model, config, workload = self.make_model_app(config_dir,
                                                      paths,
                                                      tmp_path)
        result = deserialize_looptree_output(model.run(),
                                             Context.getDefaultInstance())

        reads, writes = reads_and_writes_from_fill(result.fill,
                                                   config['mapping'],
                                                   workload)

        self.assertEqual(read_refs, get_total_accesses(reads))
        self.assertEqual(write_refs, get_total_accesses(writes))
