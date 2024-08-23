import unittest

import islpy as isl

from pytimeloop.isl.sum import sum_until_idx


class TestSum(unittest.TestCase):
    def test_sum_until_idx(self):
        pw_qp = isl.PwQPolynomial.read_from_str(
            isl.DEFAULT_CONTEXT,
            '{ [x, y] -> x + y : 0 <= y < 3}'
        )
        result = sum_until_idx(1, pw_qp)
        self.assertEqual(
            isl.PwQPolynomial.read_from_str(
                isl.DEFAULT_CONTEXT,
                '{ [x] -> 3*x + 3 }'
            ).to_str(),
            result.to_str()
        )