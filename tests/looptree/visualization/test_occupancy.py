import unittest
import islpy as isl

from pytimeloop.looptree.des import deserialize_looptree_output
from pytimeloop.looptree.visualization.occupancy import plot_occupancy_graph

from tests.util import CONFIG_DIR, TEST_TMP_DIR
from ..make_model_app import make_model_app


class TestOccupancyVisualization(unittest.TestCase):
    def test_fused(self):
        model, _, workload = make_model_app(
            CONFIG_DIR,
            [
                'looptree-test-fused.yaml',
                'cascaded_mm.workload.yaml',
                'three_level.arch.yaml'
            ],
            TEST_TMP_DIR,
            False
        )
        result = deserialize_looptree_output(model.run(),
                                             isl.DEFAULT_CONTEXT)

        plot_occupancy_graph(result, workload)