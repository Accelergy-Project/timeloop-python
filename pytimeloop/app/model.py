from bindings.config import Configurator
from pytimeloop.engine import Accelerator
from .call_utils import read_output_files

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
        def run_result_to_queue(self, q: multiprocessing.Queue):
            result = self.run()
            q.put(result)

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_result_to_queue, args=(self, q))
        p.start()
        p.join()
        return q.get()

    def run_subprocess(self, out_dir: str = None):
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

        cfg = Configurator.from_yaml_str(self.yaml_str_cfg)

        # # timeloop-model configurations
        # self.auto_bypass_on_failure = auto_bypass_on_failure
        # self.out_prefix = out_prefix
        # semi_qualified_prefix = 'timeloop-model'
        # self.out_prefix = out_dir + '/' + semi_qualified_prefix

        arch_specs = cfg.get_arch_specs()

        # Problem configuration
        workload = cfg.get_workload()
        model_logger.info("Workload configuration complete.")

        constraints = cfg.get_mapping_constraints()
        model_logger.info("Architecture configuration complete.")

        # Mapping configuration
        mapping = cfg.get_mapping()
        model_logger.info("Mapping construction complete.")

        # Validate mapping against architecture constraints
        if not constraints.satisfied_by(mapping):
            model_logger.error("Mapping violates architecture constraints.")
            raise ValueError("Mapping violates architecture constraints.")

        # Sparse optimizations
        sparse_optimizations = cfg.get_sparse_opts()

        engine = Accelerator(arch_specs)

        eval_stat = engine.evaluate(mapping, workload, sparse_optimizations)
        return eval_stat
