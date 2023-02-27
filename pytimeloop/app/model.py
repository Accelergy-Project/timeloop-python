from bindings.config import Configurator
from pytimeloop.engine import Accelerator

import os
import logging
import multiprocessing
import subprocess


class ModelApp:
    def __init__(self, yaml_str_cfg: str, log_level=logging.INFO):
        self.log_level = log_level
        self.yaml_str_cfg = yaml_str_cfg

    def run_sandboxed(self):
        def run_result_to_queue(self, q: multiprocessing.Queue):
            result = self.run()
            q.put(result)
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_result_to_queue,
                                    args=(self, q))
        p.start()
        p.join()
        return q.get()

    def run_subprocess(self):
        with open('tmp.yaml', 'w') as f:
            f.write(self.yaml_str_cfg)
        subprocess.run(['timeloop-model', 'tmp.yaml'])
        os.remove('tmp.yaml')

        path_to_stats = 'timeloop-model.stats.txt'
        stats = ''
        with open(path_to_stats, 'r') as f:
            stats += f.read()
        
        return stats

    def run(self):
        # Setup logger
        model_logger = logging.getLogger('pytimeloop.app.Model')
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
        model_logger.info('Workload configuration complete.')

        constraints = cfg.get_mapping_constraints()
        model_logger.info('Architecture configuration complete.')

        # Mapping configuration
        mapping = cfg.get_mapping()
        model_logger.info('Mapping construction complete.')

        # Validate mapping against architecture constraints
        if not constraints.satisfied_by(mapping):
            model_logger.error(
                'Mapping violates architecture constraints.')
            raise ValueError('Mapping violates architecture constraints.')

        # Sparse optimizations
        sparse_optimizations = cfg.get_sparse_opts()

        engine = Accelerator(arch_specs)

        eval_stat = engine.evaluate(mapping,
                                    workload,
                                    sparse_optimizations)
        return eval_stat
