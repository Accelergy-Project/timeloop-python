"""
Does unit/integration testing for buffer of PyTimeloop.
"""

# Imports some convenience libraries for easier unittest management.
import unittest
import typing
from pathlib import Path

# Imports the items we're testing; Engine is used to generate Buffers.
from bindings.model import Engine
from bindings.buffer import BufferLevel

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
        # Directory and path of all the config files.
        config_dir: Path = Path("01-model-conv1d-2level")
        paths: list[str] = [
            "arch/*.yaml",
            "map/conv1d-2level-os.map.yaml",
            "prob/*.yaml",
        ]
        
        # Engine that is used to generate the BufferLevel.
        engine: Engine = run_evaluation(config_dir, paths)

        print(engine)


if __name__ == "__main__":
    unittest.main()