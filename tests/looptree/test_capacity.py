import unittest
from pathlib import Path

import islpy as isl

from pytimeloop.looptree.des import deserialize_looptree_output
from pytimeloop.looptree.capacity import compute_capacity_usage
from .make_model_app import make_model_app

from tests.util import TEST_TMP_DIR


class TestCapacityAggregators(unittest.TestCase):
    def test_capacity_analysis(self):
        model, spec, workload = make_model_app(
            Path(__file__).parent.parent / 'test_configs',
            ['looptree-test-fused.yaml'],
            TEST_TMP_DIR,
            False
        )

        result = deserialize_looptree_output(model.run(), isl.DEFAULT_CONTEXT)
        capacity_usage = compute_capacity_usage(spec.mapping.nodes,
                                                result.occupancy,
                                                workload)

        self.assertEqual({0: 130, 1: 70, 2: 12}, capacity_usage)
