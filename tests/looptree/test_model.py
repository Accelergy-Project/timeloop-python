import unittest
from pathlib import Path
from itertools import starmap

import islpy as isl

from bindings.config import Config
from bindings.looptree import *
from pytimeloop.looptree.des import deserialize_looptree_output
from .make_model_app import make_model_app
from tests.util import TEST_TMP_DIR, gather_yaml_configs

class LooptreeModelAppTest(unittest.TestCase):
    def test_model_with_two_level_mm(self):
        self.check_model_app(
            Path(__file__).parent.parent / 'test_configs',
            [
                'looptree-test-fused.yaml',
                'cascaded_mm.workload.yaml',
                'three_level.arch.yaml'
            ],
            TEST_TMP_DIR
        )

    def check_model_app(
        self, config_dir, paths, tmp_path
    ):
        self.maxDiff = None
        model, _, _ = make_model_app(config_dir, paths, tmp_path)
        result = model.run()

        def compare_dim(d, dtype):
            return isinstance(d, dtype)
        def compare_dims(ds, dtypes):
            return all(starmap(compare_dim, zip(ds, dtypes)))
        def firsts(ds):
            yield from map(lambda pair: pair[0], ds)
        def seconds(ds):
            yield from map(lambda pair: pair[1], ds)

        dims = [TemporalTag, SequentialTag, TemporalTag, SpatialTag]
        self.assertTrue(compare_dims(
            list(firsts(result.ops.values()))[0],
            dims
        ))
        self.assertEqual(
            [
                "{ [i0, i1, i2, i3] -> 1 : i1 = 0 and 0 <= i0 <= 8 and 0 <= i2 <= 1 and 0 <= i3 <= 3 }",
                "{ [i0, i1, i2, i3] -> 1 : i1 = 1 and 0 <= i0 <= 8 and 0 <= i2 <= 3 and 0 <= i3 <= 7 }",
            ],
            list(seconds(result.ops.values()))
        )


class TestLooptreeOutputDeserializer(unittest.TestCase):
    def test_deserializer_with_two_level_mm(self):
        self.check_deserializer(
            Path(__file__).parent.parent / 'test_configs',
            [
                'looptree-test-fused.yaml',
                'cascaded_mm.workload.yaml',
                'three_level.arch.yaml'
            ],
            TEST_TMP_DIR
        )

    def check_deserializer(self, config_dir, paths, tmp_path):
        model, _, _ = make_model_app(config_dir, paths, tmp_path)
        result = model.run()

        des_result = deserialize_looptree_output(result, isl.DEFAULT_CONTEXT)

        self.assertEqual(
            {
                0: "{ [i0, i1, i2, i3] -> 1 : i1 = 0 and 0 <= i0 <= 8 and 0 <= i2 <= 1 and 0 <= i3 <= 3 }",
                1: "{ [i0, i1, i2, i3] -> 1 : i1 = 1 and 0 <= i0 <= 8 and 0 <= i2 <= 3 and 0 <= i3 <= 7 }",
            },
            {
                k: v.to_str() for k, (tags, v) in des_result.ops.items()
            }
        )
