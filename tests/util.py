from pathlib import Path
import glob

from pytimeloop import Config

PROJECT_DIR = Path(__file__).parent.parent
TIMELOOP_EXAMPLES_DIR  = (
    PROJECT_DIR / 'tests/timeloop-accelergy-exercises/workspace/'
    / 'exercises/2020.ispass/timeloop')
TEST_TMP_DIR = PROJECT_DIR / 'tests/tmp-files'

def load_config_patterns(input_patterns):
    yaml_str = ''
    for pattern in input_patterns:
        for fname in glob.iglob(pattern):
            with open(fname, 'r') as f:
                yaml_str += f.read()
            yaml_str += '\n'
    return Config.load_yaml(yaml_str)

def load_configs(rel_config_dir, rel_paths):
    config_dir = TIMELOOP_EXAMPLES_DIR / rel_config_dir
    paths = map(lambda p: str(config_dir / p), rel_paths)
    config = load_config_patterns(paths)
    return config
