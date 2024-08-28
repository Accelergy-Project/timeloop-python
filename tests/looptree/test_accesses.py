import unittest
from pathlib import Path

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import islpy as isl

from bindings.config import Config
from bindings.looptree import LooptreeModelApp, LooptreeWorkload
from pytimeloop.looptree.des import deserialize_looptree_output
from pytimeloop.looptree.accesses import *

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class TestLoopTreeAccess(unittest.TestCase):
    def test_accesses_with_two_level_mm_fused(self):
        self.maxDiff = None
        self.check_accesses(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR,
            {
                (0, 'Fmap1', 'Fc1'): 18,
                (0, 'Filter1', 'Fc1'): 8,
                (0, 'Filter2', 'Fc2'): 32,
                (0, 'Fmap3', 'Fc2'): 72,
                (1, 'Fmap1', 'Fc1'): 18,
                (1, 'Filter1', 'Fc1'): 72,
                (1, 'Filter2', 'Fc2'): 288,
                (1, 'Fmap3', 'Fc2'): 288,
                (2, 'Fmap2', 'Fc1'): 72,
                (2, 'Fmap2', 'Fc2'): 36,
                (3, 'Fmap1', 'Fc1'): 72,
                (3, 'Fmap2', 'Fc2'): 288
            },
            {
                (0, 'Fmap3', 'Fc2'): 72,
                (1, 'Fmap3', 'Fc2'): 288,
                (2, 'Fmap2', 'Fc1'): 72
            }
        )

    def test_accesses_with_two_level_mm_unfused(self):
        self.check_accesses(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-unfused.yaml'],
            TEST_TMP_DIR,
            {
                (0, 'Fmap1', 'Fc1'): 18,
                (0, 'Filter1', 'Fc1'): 8,
                (0, 'Filter2', 'Fc2'): 32,
                (0, 'Fmap2', 'Fc1'): 36,
                (0, 'Fmap2', 'Fc2'): 36,
                (0, 'Fmap3', 'Fc2'): 72,
                (1, 'Filter1', 'Fc1'): 72,
                (1, 'Fmap1', 'Fc1'): 72,
                (2, 'Fmap2', 'Fc1'): 72,
                (1, 'Filter2', 'Fc2'): 288,
                (1, 'Fmap3', 'Fc2'): 288,
                (2, 'Fmap2', 'Fc2'): 288
            },
            {
                (0, 'Fmap2', 'Fc1'): 36,
                (0, 'Fmap3', 'Fc2'): 72,
                (2, 'Fmap2', 'Fc1'): 72,
                (1, 'Fmap3', 'Fc2'): 288
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

        result = model.run()
        result = deserialize_looptree_output(result, isl.DEFAULT_CONTEXT)

        # print(result.fills_by_parent)
        reads, writes = reads_and_writes_from_fill_by_parent(
            result.fills_by_parent,
            config['mapping'],
            workload
        )

        self.assertEqual(read_refs, get_total_accesses(reads))
        self.assertEqual(write_refs, get_total_accesses(writes))

        reads, writes = reads_and_writes_from_fill_by_peer(
            result.fills_by_peer,
            workload
        )
        reads = get_total_accesses(reads)
        writes = get_total_accesses(writes)
        self.assertEqual(0, sum(reads.values()))
        self.assertEqual(0, sum(writes.values()))
