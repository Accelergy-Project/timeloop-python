from pathlib import Path

import islpy as isl

from bindings.config import Config
from bindings.looptree import LooptreeModelApp, LooptreeWorkload

from pytimeloop.file import gather_yaml_configs

from pytimeloop.looptree.des import deserialize_looptree_output
from pytimeloop.looptree.energy import gather_actions, compute_energy_from_actions
from pytimeloop.looptree.latency import compute_latency

from pytimeloop.timeloopfe.v4fused import Specification
from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose


def run_pytimeloop(config_dir, paths, tmp_path, bindings, call_accelergy):
    yaml_str = gather_yaml_configs(config_dir, paths)
    config = Config(yaml_str, 'yaml')
    if isinstance(tmp_path, Path):
        tmp_path = str(tmp_path)
    model = LooptreeModelApp(config, tmp_path, 'looptree-model')

    workload = LooptreeWorkload.parse_cfg(config.root['problem'])

    spec = Specification.from_yaml_files([
        str(config_dir / p) for p in paths
    ])
    if call_accelergy:
        call_accelergy_verbose(spec, tmp_path)
        spec = Specification.from_yaml_files([
            str(config_dir / p) for p in paths
        ] + [str(Path(tmp_path) / 'ERT.yaml')])

    result = deserialize_looptree_output(model.run(), isl.DEFAULT_CONTEXT)

    actions = gather_actions(result, spec.mapping, workload, bindings)
    energy = compute_energy_from_actions(actions, spec.ERT)

    latency = compute_latency(spec.mapping.nodes,
                              result.temporal_steps,
                              workload)

    return latency, energy
