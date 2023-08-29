"""
Does unit/integration testing for model::Topology of PyTimeloop, including
output testing of engine evaluations.
"""
# Imports some convenience libraries for easier unittest management.
import unittest
import typing
from pathlib import Path

# Imports the items we're testing; Engine is used to generate Topology.
from bindings.model import Engine, Topology

# Imports the test utility functions.
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
        stats: Topology.Stats = topology.stats

        # Gets all the variable names of stats.
        var_names: list[str] = {
            var_name
            for var_name in dir(stats)
            if not callable(getattr(stats, var_name))
        } - {"__doc__", "__module__"}
        # Tests we're able to access everything in Stats
        key: str
        val: typing.Any
        for key in var_names:
            # Pulls the attribute from stats.
            attr: typing.Any = getattr(stats, key)

            ## TODO:: Replace this at some point with a ground truth reference.
            print(f"{key}: {attr}")

        # Ensures that the Reset function works for stats.
        stats.reset()

        # Tests everything we can access in Stats is cleared.
        key: str
        for key in var_names:
            # Pulls the attribute from stats.
            attr: typing.Any = getattr(stats, key)

            # Tests to make sure the function is cleared
            self.assertFalse(attr)


if __name__ == "__main__":
    unittest.main()
