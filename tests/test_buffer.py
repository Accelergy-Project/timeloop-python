"""
Does unit/integration testing for the buffer namespace of PyTimeloop, including
output testing of engine evaluations.
"""

import unittest
import typing
from pathlib import Path

from bindings.model import Engine, Topology
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
        config_dir: Path = Path("01-model-conv1d-2level")
        paths: list[str] = [
            "arch/*.yaml",
            "map/conv1d-2level-os.map.yaml",
            "prob/*.yaml",
        ]

        # Runs evaluation.
        engine: Engine = run_evaluation(config_dir, paths)
        # Gets the topology.
        topology: Topology = engine.get_topology()
        # Fetches the stats of the topology.
        stats: Topology.Stats = topology.get_stats()
        
        # Tests we're able to access everything in Stats
        key: str
        for key in dir(stats):
            # Pulls the attribute from stats.
            attr: typing.Any = getattr(stats, key)

            # Makes sure if we pull a function we don't print that.
            if callable(attr):
                continue

            ## TODO:: Replace this at some point with a ground truth reference.
            print(attr)
        
        # Ensures that the Reset function works for stats.
        stats.reset()

        # Tests everything we can access in Stats is cleared.
        key: str
        for key in dir(stats):
            # Pulls the attribute from stats.
            attr: typing.Any = getattr(stats, key)

            # Makes sure if we pull a function we don't test it.
            if not callable(attr):
                continue
            
            # Tests to make sure the function is cleared
            self.assertFalse(attr)
        
if __name__ == "__main__":
    unittest.main()