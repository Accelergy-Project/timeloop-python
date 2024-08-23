import unittest
from pathlib import Path

import islpy as isl

from pytimeloop.looptree.des import deserialize_looptree_output
from .make_model_app import make_model_app
from tests.util import TEST_TMP_DIR


class TestLatency(unittest.TestCase):
    def test_latency_mm_fused(self):
        model, spec, workload = make_model_app(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR,
            False
        )
        result = deserialize_looptree_output(model.run(),
                                             isl.DEFAULT_CONTEXT)

        print(result.op_occupancy)


    def test_latency_mm_unfused(self):
        pass