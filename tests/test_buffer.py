"""
Does unit/integration testing for the buffer namespace of PyTimeloop, including
output testing of engine evaluations.
"""

import unittest
import typing

from bindings.buffer import Stats
from tests.util import run_evaluation


class StatsTest(unittest.TestCase):
    """
    @brief  Tests the stats output of buffer, making sure all accession is
            possible.
    """

    def test_accession(self) -> None:
        """Tests that we are able to read all member values of stat.

        All results are printed out or logged through unittest assert statements.

        @param self The unittest environment we're running in.
        """
        run_evaluation()
