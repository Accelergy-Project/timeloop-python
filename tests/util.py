"""
Utility functions that most tests will need.
"""

from pathlib import Path
import glob

# Imports we need to run an evaluation.
from bindings.config import Config
from bindings.problem import Workload
from bindings.model import ArchSpecs, SparseOptimizationInfo, Engine
from bindings.mapping import Mapping


# The directory of the project we're in.
PROJECT_DIR = Path(__file__).parent.parent
# The directory of the timeloop examples.
TIMELOOP_EXAMPLES_DIR = (
    PROJECT_DIR
    / "tests/timeloop-accelergy-exercises/workspace/"
    / "exercises/2020.ispass/timeloop"
)
# The directory of the test temp files.
TEST_TMP_DIR = PROJECT_DIR / "tests/tmp-files"


def gather_yaml_files(input_patterns: map) -> str:
    """Combines YAML files specified into one big string.

    @param input_patterns   The absolute paths of the files we want to access.

    @return                 The files we specified combined together.
    """
    # The return value namespace.
    yaml_str: str = ""

    # Iterates through all filepath patterns.
    pattern: str
    for pattern in input_patterns:
        # Calculates all files that match that pattern.
        fname: str
        for fname in glob.iglob(pattern):
            # Concatenates the file onto the string.
            with open(fname, "r", encoding="utf-8") as file:
                yaml_str += file.read() + "\n"
    return yaml_str


def gather_yaml_configs(config_dir: Path, rel_paths: list[str]) -> str:
    """Combines together all the yaml config files into one string.

    @param rel_config_dir   The relative directory of the configs we want to
                            load in the Timeloop examples folder.
    @param rel_paths        The relative paths of all the config files within
                            the configs directory we specified.

    @return                 The combined string of all the YAML config files.
    """
    # Constructs the absolute path of the config directory.
    config_dir: Path = config_dir
    # Constructs the absolute path of all the config files.
    paths: map = map(lambda p: str(config_dir / p), rel_paths)

    return gather_yaml_files(paths)


def run_evaluation(config_dir: Path, paths: list[str]) -> Engine:
    """Creates and runs Timeloop given a configuration directory and paths
    to the requisite YAML files.

    Outputs errors only through unittest asserts and print statements.

    @param config_dir   The directory containing the evaluation config settings.
    @param paths        The paths of all the requisite files in the directory.

    @return             The engine after it finished evaluation.
    """
    # Combined YAML string of all the config files.
    yaml_str = gather_yaml_configs(config_dir, paths)

    # Loads the YAML into Configuration settings.
    config: Config = Config(yaml_str, "yaml")
    # Pulls out the Config root node, containing all the config info.
    root: ConfigNode = config.root

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
