'''
Does unit/integration testing for the configuration/inputs of PyTimeloop.
'''

import logging
from pathlib import Path
import unittest
import typing

# import bindings
# from bindings.config import Configurator
# from bindings.config import ConfigNode

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
        """Tests getting from CompoundConfigNode when loaded by CompoundConfig.

        Returns test results only through print statements generated through the
        unittest module's assert functions.

        @param self The unit test instance, so it can access test specific vars
                    like file locations.
        """

        root: str
        directory: list[str]
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
                        # Load the truth we're using for comparison.
                        truth: dict = self.yaml.safe_load(file)
                        # Close the stream.
                        print(filename, truth)
                        file.close()


if __name__ == "__main__":
    unittest.main()