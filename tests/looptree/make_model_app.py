from pathlib import Path

from tests.util import gather_yaml_configs, TEST_TMP_DIR

from bindings.config import Config
from bindings.looptree import LooptreeModelApp, LooptreeWorkload
from pytimeloop.timeloopfe.v4fused import Specification
from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose


def make_model_app(config_dir: Path, paths: list, tmp_path, call_accelergy: bool = True):
    yaml_str = gather_yaml_configs(config_dir, paths)
    config = Config(yaml_str, 'yaml')
    model = LooptreeModelApp(config, str(tmp_path), 'looptree-model')

    spec = Specification.from_yaml_files([
        str(config_dir / p) for p in paths
    ])

    workload = LooptreeWorkload.parse_cfg(config.root['problem'])

    if call_accelergy:
        call_accelergy_verbose(spec, TEST_TMP_DIR)
        spec = Specification.from_yaml_files([
            str(config_dir / p) for p in paths
        ] + [str(TEST_TMP_DIR / 'ERT.yaml')])

    return model, spec, workload
