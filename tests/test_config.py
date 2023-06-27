'''
Does unit/integration testing for the configuration/inputs of PyTimeloop.
'''

import logging
from pathlib import Path
import unittest
import typing

import bindings
from bindings.config import Configurator
from bindings.config import Config
from bindings.config import ConfigNode

# from .util import TEST_TMP_DIR, gather_yaml_configs


# class ConfigTest(unittest.TestCase):
#     def test_config(self):
#         CONFIG_DIR = Path('01-model-conv1d-2level')
#         PATHS = ['arch/*.yaml',
#                  'map/conv1d-2level-os.map.yaml',
#                  'prob/*.yaml']
#         yaml_str = gather_yaml_configs(CONFIG_DIR, PATHS)
#         configurator = Configurator.from_yaml_str(yaml_str)

#         self.arch_specs = configurator.get_arch_specs()
#         self.workload = configurator.get_workload()
#         self.mapping = configurator.get_mapping()
#         self.sparse_opts = configurator.get_sparse_opts()

## @var The testing seed.
seed: int = 42

class CompoundConfigNodeTest(unittest.TestCase):
    '''Tests the CompoundConfigNode class's reads and writes.
    '''

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
                    
    def test_accession(self) -> None:
        '''Tests getting from CompoundConfigNode when loaded by CompoundConfig.

        Returns test results only through print statements generated through the
        unittest module's assert functions.

        @param self The unit test instance, so it can access test specific vars
                    like file locations.
        '''
        print("Testing Accessions:\n-----------")

        root: str
        _: list[str]
        file: list[str]
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
                        data:str = file.read()
                        # Load the truth we're using for comparison.
                        truth: dict = self.yaml.safe_load(data)
                        # Load the truth into Config.
                        compound_config: Config = Config(data, "yaml")

                    # Pulls the Node (dict structure analog) from Config.
                    node: ConfigNode = compound_config.getRoot()
                    # Starts recursive test.
                    self.check_node(truth, node)
                    
    def test_setting_fuzz(self) -> None:
        '''Tests setting to CompoundConfigNode when provided a random input.

        Returns test results only through print statements generated through the
        unittest module's assert functions.

        @param self The unit test instance this test belongs to.
        '''
        print("\n\n\nTesting Setters (Fuzz):\n-----------")

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
            match self.rng.randint(0, 4):
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
                    if not key in truth or not isinstance(truth[key], list):
                        truth[key] = []
                        root[key] = None
                    
                    truth[key].append(val)
                    root[key].append(val)
                # Map, we will add to Map if Map already at key.
                case 4:
                    # If not a map here, make a map here.
                    if not key in truth or not isinstance(truth[key], dict):
                        truth[key] = {}
                        root[key] = None
                    
                    # Generates the subkey.
                    subkey: int = self.rng.randint(self.min_rng, self.max_rng / 100)

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


if __name__ == "__main__":
    unittest.main()
