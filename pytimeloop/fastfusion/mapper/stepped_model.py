from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from pytimeloop.looptree.energy import gather_actions, compute_energy_from_actions
from pytimeloop.looptree.fastmodel import run_fastmodel

from pytimeloop.timeloopfe.v4 import Ert
from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose


@dataclass
class Stats:
    latency: float = 0
    energy: float = 0
    spatial: list = None
    capacity: dict = None

    def __add__(self, other: 'Stats'):
        stats = Stats()
        stats.energy = self.energy + other.energy
        stats.latency = max(self.latency, other.latency)
        return stats


class SteppedModelState:
    def __init__(self):
        self.mapping = []
        self.mapping_of_interest = []
        self.id_of_einsum_to_eval = None


class SteppedModel:
    def __init__(self, config, spec, bindings, workload, analyzer):
        self.config = config
        self.spec = spec
        self.bindings = bindings
        self.workload = workload
        self.analyzer = analyzer
        self.tensor_id_to_name = workload.data_space_id_to_name()
        self.dimension_id_to_name = workload.dimension_id_to_name()
        self.ert = None
        self.eval_count = 0

    def call_accelergy(self, tmp_path: Path):
        if isinstance(tmp_path, Path):
            tmp_path = str(tmp_path)
        call_accelergy_verbose(self.spec, tmp_path)
        ert_dict = yaml.load(Path(tmp_path) / 'ERT.yaml')
        self.ert = Ert(ert_dict['ERT'])

    def initialize(self, state, level, id_of_einsum_to_eval, retained_tensors):
        state.mapping.append({
            'type': 'storage',
            'target': level,
            'dspace': [self.tensor_id_to_name[tensor_id]
                       for tensor_id in retained_tensors]
        })
        branches = []
        state.mapping.append({
            'type': 'sequential',
            'branches': branches
        })
        for einsum_id, einsum_name in self.workload.einsum_id_to_name().items():
            if einsum_id != id_of_einsum_to_eval:
                branches.append([{
                    'type': 'compute',
                    'target': len(self.bindings)-1,
                    'einsum': einsum_name,
                    'incomplete': True
                }])
            else:
                branches.append(state.mapping_of_interest)

        state.id_of_einsum_to_eval = id_of_einsum_to_eval


    def add_storage(self, state, level, temporal_loops, spatial_loops, retained_tensors):
        self.add_temporal_and_spatial_loops(state, temporal_loops, spatial_loops)
        state.mapping_of_interest.append({
            'type': 'storage',
            'target': level,
            'dspace': [self.tensor_id_to_name[tensor_id]
                       for tensor_id in retained_tensors]
        })

    def add_compute(self, state, level, einsum_name, temporal_loops, spatial_loops):
        self.add_temporal_and_spatial_loops(state, temporal_loops, spatial_loops)
        state.mapping_of_interest.append({
            'type': 'compute',
            'target': level,
            'einsum': einsum_name
        })

    def run(self, state):
        # arch_workload_cfg = str(self.config.root)
        # mapping_strbuf = io.StringIO()
        # yaml.dump({'mapping': {'type': "fused", 'nodes': self.mapping}},
        #           mapping_strbuf)
        # mapping_cfg = mapping_strbuf.getvalue()
        # config_str = mapping_cfg + arch_workload_cfg
        # config = Config(config_str, 'yaml')

        # model = LooptreeModelApp(config)
        self.eval_count += 1
        result = run_fastmodel({'nodes': state.mapping},
                               state.id_of_einsum_to_eval,
                               self.workload,
                               self.analyzer)

        actions = gather_actions(result,
                                 {'type': 'fused', 'nodes': state.mapping},
                                 self.workload,
                                 self.bindings)
        energy = compute_energy_from_actions(actions, self.ert)
        energy = sum(energy.values())

        # latency = compute_latency(state.mapping,
        #                           result.temporal_steps,
        #                           self.workload)

        stats = Stats()
        stats.energy = energy
        stats.latency = result.temporal_steps[state.id_of_einsum_to_eval]
        stats.spatial = result.fanout
        stats.capacity = result.occupancy

        return stats

    def add_temporal_and_spatial_loops(self, state, temporal_loops, spatial_loops):
        for rank, shape in temporal_loops:
            state.mapping_of_interest.append({
                'type': 'temporal',
                'rank': self.dimension_id_to_name[rank],
                'tile_shape': shape
            })
        for spatial_idx, loops in enumerate(spatial_loops):
            for rank, shape in loops:
                state.mapping_of_interest.append({
                    'type': 'spatial',
                    'spatial': spatial_idx,
                    'rank': self.dimension_id_to_name[rank],
                    'tile_shape': shape
                })