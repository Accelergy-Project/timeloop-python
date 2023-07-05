"""
Does unit/integration testing for the configuration/inputs of PyTimeloop.
"""

from pathlib import Path
import unittest
import typing

from util import TEST_TMP_DIR, gather_yaml_configs

from bindings.config import Config
from bindings.config import ConfigNode

from bindings.problem import Workload
from bindings.model import ArchSpecs, SparseOptimizationInfo, Engine
from bindings.mapping import Mapping


class ConfigTest(unittest.TestCase):
    """
    @brief  Tests the CompoundConfig bindings, ensuring a Timeloop workload can
            run solely with Python.
    """

    def run_evaluation(self, config_dir: Path, paths: list[str]) -> Engine:
        """Creates and runs Timeloop given a configuration directory and paths
        to the requisite YAML files.

        Outputs errors only through unittest asserts and print statements.

        @param self         The testing environment.
        @param config_dir   The directory containing the evaluation config settings.
        @param paths        The paths of all the requisite files in the directory.

        @return             The engine after it finished evaluation.
        """
        # Combined YAML string of all the config files.
        yaml_str = gather_yaml_configs(config_dir, paths)

        # Loads the YAML into Configuration settings.
        config: Config = Config(yaml_str, "yaml")
        # Pulls out the Config root node, containing all the config info.
        root: ConfigNode = config.getRoot()

        # Creates the workload specified by root.
        workload: Workload = Workload(root["problem"])
        # Creates the architecture specified by root.
        arch_specs: ArchSpecs = ArchSpecs(
            root["architecture"], "sparse_optimizations" in root
        )

        # Does accelergy load-ins if present.
        if "ERT" in root:
            arch_specs.parse_accelergy_ert(root["ERT"])
        if "ART" in root:
            arch_specs.parse_accelergy_art(root["ART"])

        # Creates the mapping off of the specifications and workload.
        mapping: Mapping = Mapping(root["mapping"], arch_specs, workload)
        # Creates SparseOptimizations off of settings.
        sparse_info: SparseOptimizationInfo = SparseOptimizationInfo(root, arch_specs)

        # Creates the evaluation engine with the specs.
        engine: Engine = Engine(arch_specs)
        # Runs the evaluator.
        engine.evaluate(mapping, workload, sparse_info)

        return engine

    def test_config_basic(self):
        """Tests that a loaded in Config can make a valid mapping.

        Basic test that Python can run a Timeloop configuration from start to
        finish off of pre-loaded files. Outputs are only through unittest asserts
        and print statements.

        @param self The testing unit environment.
        """
        print("\n\n\nTesting Integration:\n" + "-" * 5)
        # Directory and path of all the config files.
        config_dir = Path("01-model-conv1d-2level")
        paths = ["arch/*.yaml", "map/conv1d-2level-os.map.yaml", "prob/*.yaml"]

        print(self.run_evaluation(config_dir, paths).pretty_print_stats())

    # def test_multiple_workloads(self):
    #     '''Tests for any errors when there exist multiple Timeloop Workload
    #     instances.

    #     Errors are printed out through unittest asserts and print statements.

    #     @param self The testing suite environment.
    #     '''
    #     print("\n\n\nTesting Multiple Workloads Existing:\n" + '-'*5)

    #     # Directory and path of all the config files.
    #     CONFIG_DIR = Path('01-model-conv1d-2level')
    #     PATHS = ['arch/*.yaml',
    #              'map/conv1d-2level-os.map.yaml',
    #              'prob/*.yaml']
    #     self.run_evaluation(CONFIG_DIR, PATHS)

    #     CONFIG_DIR = Path('01-model-conv1d-1level')
    #     PATHS = ['arch/*.yaml',
    #              'map/conv1d-2level-os.map.yaml',
    #              'prob/*.yaml']
    #     self.run_evaluation(CONFIG_DIR, PATHS)


## @var The testing seed.
seed: int = 42


class CompoundConfigNodeTest(unittest.TestCase):
    """Tests the CompoundConfigNode class's reads and writes."""

    ## @var A parsed YAML file is used as the ground truth for our test cases.
    import yaml

    ## @var The general file location of the tests we're going to run.
    import os

    ## @var The entropy library we're using.
    import random

    ## @var The seeded RNG we're using.
    rng: random.Random = random.Random(seed)
    ## @var The number of cycles we expect fuzz tests to perform.
    tests: int = 1000
    ## @var The min value of the rng.
    min_rng: int = 0
    ## @var The max value of the rng.
    max_rng: int = tests // 10

    ## @var The location of all our YAML files.
    root: str = (
        os.path.dirname(__file__)
        + "/timeloop-accelergy-exercises/"
        + "workspace/exercises/2020.ispass/timeloop"
    )

    def check_node(self, truth: typing.Union[list, dict], node: ConfigNode):
        """Checks that a node is equal to its equivalent truth. Returns nothing.
        Values only arise from unittest printouts.

        @param self     The test case module.
        @param truth    The truth value we're referencing the Node against.
        @param node     The node we're checking for correct reads.
        """

        # Defines different behavior is truth is a dict vs list.
        truth_keys: list = None
        # List truth.
        if isinstance(truth, (list, tuple)):
            truth_keys = range(len(truth))
        # Dict truth.
        else:
            truth_keys = truth.keys()

        # Goes through all the keys in truth.
        key: typing.Union[int, str]
        for key in truth_keys:
            # If value is a scalar, compare.
            if isinstance(truth[key], (bool, float, int, str, type(None))):
                self.assertEqual(truth[key], node[key].resolve())
            # Otherwise, it is a node, so recurse.
            else:
                self.check_node(truth[key], node[key])

    def rummage_files(self, func: callable) -> None:
        """Goes through all files in the testing suite and runs a given test fxn
        on the files, passing in the files as a string.

        All outputs are given through the unittest asserts and print statements.

        @param self The testing suite environment.
        @param func The testing function we want to do over all files in the
                    testing suite environment.
        """
        root: str
        _: list[str]
        files: list[str]
        # Goes through all the files in the testing directory.
        for root, _, files in self.os.walk(self.root, topdown=False):
            # Goes though all files.
            for file in files:
                # Only runs test if we pulled in a YAML file.
                if file.endswith(".yaml"):
                    # Constructs the file name.
                    filename: str = self.os.path.join(root, file)

                    # Open the file.
                    with open(filename, "r") as file:
                        # Reads the data as a string.
                        data: str = file.read()
                        # Passes parsed data into testing fxn.
                        func(data)

    def test_accession(self) -> None:
        """Tests getting from CompoundConfigNode when loaded by CompoundConfig.

        Returns test results only through print statements generated through the
        unittest module's assert functions.

        @param self The unit test instance, so it can access test specific vars
                    like file locations.
        """
        print("Testing Accessions:\n" + "-" * 5)

        def accession_test(data: dict) -> None:
            """Given a string of a canonical YAML file, test that all accesses
            are possible.

            All errors are reported through print statements and the unittest
            assert functions.

            @param data The data in the canonical YAML file.
            """
            # Load the truth we're using for comparison.
            truth: dict = self.yaml.safe_load(data)
            # Load the truth into Config.
            compound_config: Config = Config(data, "yaml")
            # Pulls the Node (dict structure analog) from Config.
            node: ConfigNode = compound_config.getRoot()
            # Starts recursive test.
            self.check_node(truth, node)

        self.rummage_files(accession_test)

    def test_setting_fuzz(self) -> None:
        """Tests setting to CompoundConfigNode when provided a random input.

        Returns test results only through print statements generated through the
        unittest module's assert functions.

        @param self The unit test instance this test belongs to.
        """
        print("\n\n\nTesting Setters (Fuzz):\n" + "-" * 5)

        # Reference truth value.
        truth: dict = {}
        # Root CCN we're working with.
        root: ConfigNode = ConfigNode()

        # Runs this amount of fuzz tests.
        for _ in range(self.tests):
            # Determines the key we want to insert to.
            key: int = self.rng.randint(self.min_rng, self.max_rng)
            # The rng value we wish to add.
            val: float = self.rng.random()

            # Determines what type we want to insert.
            yaml_type: int = self.rng.randint(0, 4)
            match yaml_type:
                # Null type.
                case 1:
                    truth[key] = None
                    root[key] = None
                # Scalar, we will assume floats for simplicity.
                case 2:
                    truth[key] = val
                    root[key] = val
                # Sequence, we will append if list already at key.
                case 3:
                    # If not a list here, make a list here.
                    if (key not in truth) or not isinstance(truth[key], list):
                        truth[key] = []
                        root[key] = ConfigNode()

                    truth[key].append(val)
                    root[key].append(val)
                # Map, we will add to Map if Map already at key.
                case 4:
                    # If not a map here, make a map here.
                    if (key not in truth) or not isinstance(truth[key], dict):
                        truth[key] = {}
                        root[key] = ConfigNode()

                    # Generates the subkey.
                    subkey: int = self.rng.randint(self.min_rng, self.max_rng // 10)

                    # Does assignment to sub Map.
                    truth[key][subkey] = val
                    root[key][subkey] = val
                # Undefined, not implemented but we might in the future.
                case 0:
                    pass

            # Checks after every write.
            self.check_node(truth, root)

        # And at the end, just in case.
        self.check_node(truth, root)

    def test_replication(self) -> None:
        """Tests ability to write in a Config from scratch by duplicating existing
        Configs that are known goods.

        Error values are returned only through the unittest library's print
        statements/asserts.

        @param self The testing suite environment we're running in.
        """
        print("\n\n\nTest Replication:\n" + "-" * 5)

        def replication_test(file: str) -> None:
            """Given a canonical YAML string, test if the string can be written
            entirely in Python.

            Error values are returned only through the unittest library's print
            statements/asserts.

            @param file The canonical YAML string.
            """
            # Loads in the reference value
            truth: dict = self.yaml.safe_load(file)
            # Creates the testing CompoundConfigNode
            root: ConfigNode = ConfigNode()

            def dupe_level(truth: typing.Union[dict, list], node: ConfigNode) -> None:
                """Deep copies this level of the tree. Recurses to copy the nested
                levels.

                We do not return anything; all copies go into node.

                @param truth    The values we want to copy.
                @param node     The place we want to copy the values to.
                """
                match (truth):
                    # Dictionary copy.
                    case dict():
                        # Goes through all keys in truth.
                        key: str
                        for key in truth:
                            # If this is a nested thing, recurse.
                            if isinstance(truth[key], (dict, list)):
                                # Instantiate the key.
                                node[key] = ConfigNode()
                                # Recurses down one layer for copying.
                                dupe_level(truth[key], node[key])
                            # Otherwise, directly set.
                            else:
                                node[key] = truth[key]
                    # List copy.
                    case list():
                        # Goes through all elements.
                        index: int
                        for index in range(len(truth)):
                            # Instantiates the value at index.
                            node.append(ConfigNode())
                            # If this is a nested thing, recurse.
                            if isinstance(truth[index], (dict, list)):
                                # Recurses down one layer for copying.
                                dupe_level(truth[index], node[index])
                            # Otherwise, directly set.
                            else:
                                node[index] = truth[index]
                    # Scalar copy.
                    case _:
                        node = truth

                # Checks equality at the end of duplicaiton.
                self.check_node(truth, node)

            dupe_level(truth, root)
            # Checks outside of recursion just in case.
            self.check_node(truth, root)

        # Does the replication test over all files we want in the tester.
        self.rummage_files(replication_test)


if __name__ == "__main__":
    unittest.main()
