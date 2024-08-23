import unittest

import islpy as isl

from pytimeloop.isl.reduction import make_reduction_map


class TestReduction(unittest.TestCase):
    def test_reduction_map(self):
        test_space = isl.BasicSet.read_from_str(
            isl.DEFAULT_CONTEXT,
            '{ [x, y, z] }'
        ).space
        reduction_map = make_reduction_map(test_space, 1, 1)
        self.assertEqual(
            isl.BasicMap.read_from_str(
                isl.DEFAULT_CONTEXT,
                '{ [x, z] -> [x, y, z] }'
            ),
            reduction_map
        )
