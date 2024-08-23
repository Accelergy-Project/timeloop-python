import unittest

import islpy as isl

from pytimeloop.isl.qpolynomial import *


class TestFromPwQPolynomialFold(unittest.TestCase):
    def test_pw_qpolynomial_fold_1(self):
        ref_pw_qp = isl.PwQPolynomial.read_from_str(
            isl.DEFAULT_CONTEXT,
            '{ [x] -> x*2 : 0 <= x < 3 }'
        )
        test_pw_qp_fold = isl.PwQPolynomialFold.from_pw_qpolynomial(
            isl.fold.max,
            ref_pw_qp
        )
        result = from_pw_qpolynomial_fold(test_pw_qp_fold)

        self.assertEqual(ref_pw_qp.to_str(), result.to_str())

    def test_pw_qpolynomial_fold_2(self):
        ref_pw_qp = isl.PwQPolynomial.read_from_str(
            isl.DEFAULT_CONTEXT,
            '{ [x] -> x*x : 0 <= x < 3 }'
        )
        test_pw_qp_fold = isl.PwQPolynomialFold.from_pw_qpolynomial(
            isl.fold.max,
            ref_pw_qp
        )
        result = from_pw_qpolynomial_fold(test_pw_qp_fold)

        self.assertEqual(ref_pw_qp.to_str(), result.to_str())
