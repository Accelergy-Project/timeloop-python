import unittest
import typing

from bindings.buffer import Stats
from util import gather_yaml_configs


class StatsTest(unittest.TestCase):
    def evaluate_workload(self, config_dir: Path, paths: list[str]) -> Engine:
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
