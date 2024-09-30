from pathlib import Path

from bindings.config import Config

from pytimeloop.timeloopfe.v4fused import Specification

from tests.util import gather_yaml_configs, CONFIG_DIR

class LoadConfigMixin:
    @staticmethod
    def load_config(paths):
        yaml_str = gather_yaml_configs(CONFIG_DIR, paths)

        config = Config(yaml_str, 'yaml')

        spec = Specification.from_yaml_files([CONFIG_DIR / p for p in paths])

        return config, spec
