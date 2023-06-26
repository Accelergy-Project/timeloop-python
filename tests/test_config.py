'''
Does unit/integration testing for the configuration/inputs of PyTimeloop.
'''

import logging
from pathlib import Path
import unittest
import typing

import bindings
# from bindings.config import Configurator
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


class CompoundConfigNodeTest(unittest.TestCase):
    '''Tests the CompoundConfigNode class's reads and writes.
    '''

    ## @var A parsed YAML file is used as the ground truth for our test cases.
    import yaml

    ## @var The general file location of the tests we're going to run.
    import os

    ## @var The location of all our YAML files.
    root: str = (
        os.path.dirname(__file__)
        + "/timeloop-accelergy-exercises/"
        + "workspace/exercises/2020.ispass/timeloop"
    )

    def test_accession(self) -> None:
        '''Tests getting from CompoundConfigNode when loaded by CompoundConfig.

        Returns test results only through print statements generated through the
        unittest module's assert functions.

        @param self The unit test instance, so it can access test specific vars
                    like file locations.
        '''
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

                    def check_node(truth: typing.Union[list, dict], node: ConfigNode):
                        """
                        Checks that a node is equal to its equivalent truth.
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
                                print(node[key])
                                self.assertEqual(truth[key], node[key])
                            # Otherwise, it is a node, so recurse.
                            else:
                                check_node(truth[key], node[key])
                    
                    check_node(truth, node)
                    




if __name__ == "__main__":
    unittest.main()
