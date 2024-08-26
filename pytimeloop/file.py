import glob
from pathlib import Path


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