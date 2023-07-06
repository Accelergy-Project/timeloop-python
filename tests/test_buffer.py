"""
Does unit/integration testing for buffer of PyTimeloop.
"""

# Imports some convenience libraries for easier unittest management.
import unittest
import typing
from pathlib import Path

# Imports the items we're testing; Engine is used to generate Buffers.
from bindings.model import Engine, Buffer

# Imports the test utility functions.
from tests.util import run_evaluation

class StatsTest(unittest.TestCase):
    """
    Tests that we are able to access BufferLevel::Stats in Python.
    """
    def test_accession(self) -> None:
        """Tests we are able to access all instance variables of BufferLevel.Stats.

        All test results will be printed out or logged through unittest assert.
        Nothing is returned.

        @param self The testing environment.
        """
        pass