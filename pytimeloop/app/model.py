from bindings.config import Config
from bindings.mapping import ArchConstraints, ArchProperties
from pytimeloop.engine import Accelerator
from .call_utils import read_output_files
from pytimeloop.problem import Workload
from pytimeloop.mapping import Mapping
from pytimeloop.model import ArchSpecs, SparseOptimizationInfo

import os
import logging
import multiprocessing
import subprocess

logger = logging.getLogger(__name__)


class ModelApp:
    def __init__(
        self, yaml_str_cfg: str, log_level=logging.INFO, default_out_dir: str = "."
    ):
        self.log_level = log_level
        self.yaml_str_cfg = yaml_str_cfg
        self._default_out_dir = default_out_dir

    def run_sandboxed(self):
        # TODO: may be outdated
        # TODO: this may not be needed anymore
        def run_result_to_queue(self, q: multiprocessing.Queue):
            result = self.run()
            q.put(result)

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_result_to_queue, args=(self, q))
        p.start()
        p.join()
        return q.get()

    def run_subprocess(self, out_dir: str = None):
        # TODO: may be outdated
        # TODO: this may not be needed anymore
        out_dir = self._default_out_dir if out_dir is None else out_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "inputs"), exist_ok=True)
        PATH_TO_TMP_INPUT = os.path.join(out_dir, "inputs", "input.yaml")
        with open(PATH_TO_TMP_INPUT, "w") as f:
            f.write(self.yaml_str_cfg)
        PATH_TO_TMP_INPUT = os.path.abspath(os.path.realpath(PATH_TO_TMP_INPUT))
        out_dir = os.path.abspath(os.path.realpath(out_dir))
        cmd = ["timeloop-model", PATH_TO_TMP_INPUT, "-o", out_dir]
        logger.info(f'Running Timeloop with command: {" ".join(cmd)}')
        result = subprocess.run(cmd, cwd=out_dir, env=os.environ, capture_output=True)
        stats, mapping = read_output_files(
            result,
            out_dir,
            "timeloop-model",
            "timeloop-model.stats.txt",
            "timeloop-model.map.txt",
        )
        return stats, mapping

    def run(self):
        # Setup logger
        model_logger = logging.getLogger("pytimeloop.app.Model")
        model_logger.setLevel(self.log_level)

        cfg = Config(self.yaml_str_cfg, 'yaml')

        # Problem configuration
        workload = Workload(cfg.root['problem'])
        model_logger.info("Workload configured.")

        is_sparse_topology = 'sparse_optimizations' in cfg.root

        arch_specs = ArchSpecs(cfg.root['architecture'], is_sparse_topology)
        model_logger.info("Arch specifications configured.")

        # Mapping configuration
        mapping = Mapping(cfg.root['mapping'], arch_specs, workload)
        model_logger.info("Mapping configured.")

        # Validate mapping against architecture constraints
        arch_props = ArchProperties(arch_specs)
        arch_constrains_cfg = Config.ConfigNode()
        if 'constraints' in cfg.root['architecture']:
            arch_constrains_cfg = cfg.root['architecture']['constraints']
        elif 'architecture_constraints' in cfg.root:
            arch_constrains_cfg = cfg.root['architecture_constraints']
        constraints = ArchConstraints(arch_props,
                                      workload,
                                      arch_constrains_cfg)
        if not constraints.satisfied_by(mapping):
            model_logger.error("Mapping violates architecture constraints.")
            raise ValueError("Mapping violates architecture constraints.")


        # Sparse optimizations
        if 'sparse_optimizations' in cfg.root:
            sparse_optimizations = SparseOptimizationInfo(
                cfg.root['sparse_optimizations'],
                arch_specs
            )
        else:
            sparse_optimizations = SparseOptimizationInfo(
                Config.ConfigNode(),
                arch_specs
            )

        engine = Accelerator(arch_specs)

        eval_stat = engine.evaluate(mapping, workload, sparse_optimizations)
        return eval_stat
