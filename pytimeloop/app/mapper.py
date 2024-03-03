from enum import Enum
import os
import logging
import subprocess
from .call_utils import read_output_files

logger = logging.getLogger(__name__)


class Betterness(Enum):
    BETTER = 1
    SLIGHTLY_BETTER = 2
    WORSE = -1
    SLIGHTLY_WORSE = -2


class SearchTask:
    def __init__(self, task_id, mapping, only_bypass_changed):
        self.task_id = task_id
        self.mapping = mapping
        self.only_bypass = only_bypass_changed


class MapperApp:
    def __init__(
        self, yaml_str_cfg: str, log_level=logging.INFO, default_out_dir: str = "."
    ):
        self.log_level = log_level
        self.yaml_str_cfg = yaml_str_cfg
        self._default_out_dir = default_out_dir

    def run_subprocess(self, out_dir: str = None):
        out_dir = self._default_out_dir if out_dir is None else out_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "inputs"), exist_ok=True)
        PATH_TO_TMP_INPUT = os.path.join(out_dir, "inputs", "input.yaml")
        with open(PATH_TO_TMP_INPUT, "w") as f:
            f.write(self.yaml_str_cfg)
        PATH_TO_TMP_INPUT = os.path.abspath(os.path.realpath(PATH_TO_TMP_INPUT))
        out_dir = os.path.abspath(os.path.realpath(out_dir))
        cmd = ["timeloop-mapper", PATH_TO_TMP_INPUT, "-o", out_dir]
        logger.info(f'Running Timeloop with command: {" ".join(cmd)}')
        result = subprocess.run(cmd, cwd=out_dir, env=os.environ, capture_output=True)
        stats, mapping = read_output_files(
            result,
            out_dir,
            "timeloop-mapper",
            "timeloop-mapper.stats.txt",
            "timeloop-mapper.map.txt",
        )
        return stats, mapping

    def run(self):
        raise NotImplementedError("Disabled for now")


# class MapperApp:
#     def __init__(self, cfg, out_dir: str, auto_bypass_on_failure=False,
#                  out_prefix='', log_level=logging.INFO):
#         # Setup logger
#         self.log_level = log_level
#         self.logger = logging.getLogger('pytimeloop.app.Mapper')
#         self.logger.setLevel(log_level)

#         # timeloop-mapper configurations
#         self.auto_bypass_on_failure = auto_bypass_on_failure
#         self.out_prefix = out_prefix
#         semi_qualified_prefix = 'timeloop-mapper'
#         self.out_prefix = out_dir + '/' + semi_qualified_prefix

#         # Architecture configuration
#         self.arch_specs = ArchSpecs(cfg['architecture'])
#         self.arch_specs.generate_tables(
#             cfg, semi_qualified_prefix, out_dir, self.out_prefix, log_level)

#         # Problem configuration
#         self.workload = Workload(cfg['problem'])
#         self.logger.info('Problem configuration complete.')

#         self.arch_props = ArchProperties(self.arch_specs)

#         # Mapper configuration
#         mapper_cfg = cfg['mapper']
#         self.num_threads = multiprocessing.cpu_count()
#         if 'num_threads' in mapper_cfg:
#             self.num_threads = mapper_cfg['num_threads']
#         self.logger.info('Using threads = %d', self.num_threads)

#         self.metrics = []
#         if 'optimization_metric' in mapper_cfg:
#             self.metrics = mapper_cfg['optimization_metric']
#         elif 'optimization_metrics' in mapper_cfg:
#             self.metrics = mapper_cfg['optimization_metrics']
#         else:
#             self.metrics = ['edp']
#         self.metrics = list(self.metrics)

#         # Search size (divide between threads)
#         self.search_size = 0
#         if 'search_size' in mapper_cfg:
#             self.search_size = mapper_cfg['search_size']
#         if 'search_size' in mapper_cfg:  # backwards compat.
#             self.search_size = mapper_cfg['search_size']
#         if self.search_size > 0:
#             self.search_size = 1 + (self.search_size - 1) // self.num_threads

#         # Num. of consecutive invalid mappings to trigger termination
#         self.timeout = 1000
#         if 'timeout' in mapper_cfg:
#             self.timeout = mapper_cfg['timeout']
#         if 'heartbeat' in mapper_cfg:  # backwards compat.
#             self.timeout = mapper_cfg['heartbeat']

#         # Number of suboptimal valid mappings to trigger victory
#         self.victory_condition = 500
#         if 'victory_condition' in mapper_cfg:
#             self.victory_condition = mapper_cfg['victory_condition']

#         # Inter-thread sync interval
#         self.sync_interval = 0
#         if 'sync_interval' in mapper_cfg:
#             self.sync_interval = mapper_cfg['sync_interval']

#         # Misc.
#         self.log_stats = False
#         if 'log_stats' in mapper_cfg:
#             self.log_stats = mapper_cfg['log_stats']

#         self.log_suboptimal = False
#         if 'log_suboptimal' in mapper_cfg:
#             self.log_suboptimal = mapper_cfg['log_suboptimal']
#         if 'log_all' in mapper_cfg:  # backwards compat.
#             self.log_suboptimal = mapper_cfg['log_all']

#         self.live_status = False
#         if 'live_status' in mapper_cfg:
#             self.live_status = mapper_cfg['live_status']

#         self.diagnostics_on = False
#         if 'diagnostics' in mapper_cfg:
#             self.live_status = mapper_cfg['diagnostics']

#         self.penalize_consecutive_bypass_fails = False
#         if 'penalize_consecutive_bypass_fails' in mapper_cfg:
#             self.penalize_consecutive_bypass_fails = \
#                 mapper_cfg['penalize_consecutive_bypass_fails']

#         self.emit_whoop_nest = False
#         if 'emit_whoop_nest' in mapper_cfg:
#             self.emit_whoop_nest = mapper_cfg['emit_whoop_nest']

#         self.logger.info('Mapper configurations complete')

#         # Architecture constraints
#         self.constraints = ArchConstraints(
#             self.arch_props, self.workload, cfg['architecture_constraints'])
#         self.logger.info('Architecture configuration complete.')

#         # Mapspace configuration
#         mapspace_cfg = Config()
#         if 'mapspace' in cfg:
#             mapspace_cfg = cfg['mapspace']
#         elif 'mapspace_constraints' in cfg:
#             mapspace_cfg = cfg['mapspace_constraints']

#         self.mapspace = MapSpace.parse_and_construct(
#             mapspace_cfg, cfg['architecture_constraints'], self.arch_specs,
#             self.workload, log_level)
#         self.split_mapspaces = self.mapspace.split(self.num_threads)
#         self.logger.info('Mapspace construction complete.')

#         # Search configuration
#         self.search = []
#         for t in range(self.num_threads):
#             self.search.append(SearchAlgorithm.parse_and_construct(
#                 mapper_cfg, self.split_mapspaces[t], t))
#         self.logger.info('Search configuration complete.')

#         # Sparse optimizations
#         if 'sparse_optimizations' in cfg:
#             sparse_opt_cfg = cfg['sparse_optimizations']
#         else:
#             sparse_opt_cfg = Config()
#         self.sparse_optimizations = SparseOptimizationInfo(
#             sparse_opt_cfg, self.arch_specs)

#         # TODO: characterize workload on whether it has metadata

#     def run(self):
#         mapper = CoupledMapper(self.arch_specs, self.workload,
#                                [(self.split_mapspaces[i], self.search[i])
#                                 for i in range(len(self.search))],
#                                self.sparse_optimizations, self.metrics,
#                                self.search_size, self.timeout,
#                                self.victory_condition,
#                                self.penalize_consecutive_bypass_fails)

#         mapping, eval_stats = mapper.run()

#         return eval_stats, mapping
