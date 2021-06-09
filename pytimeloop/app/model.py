from bindings import ArchProperties
from pytimeloop.config import Config
from pytimeloop.engine import Accelerator
from pytimeloop.model import ArchSpecs
from pytimeloop.mapping import ArchConstraints, Mapping
from pytimeloop.problem import Workload

import sys
import logging


class Model:
    def __init__(self, cfg: Config, out_dir: str, auto_bypass_on_failure=False,
                 out_prefix='', log_level=logging.INFO):
        # Setup logger
        self.log_level = log_level
        self.model_logger = logging.getLogger('pytimeloop.app.Model')
        self.model_logger.setLevel(log_level)

        # timeloop-model configurations
        self.auto_bypass_on_failure = auto_bypass_on_failure
        self.out_prefix = out_prefix
        semi_qualified_prefix = 'timeloop-model'
        self.out_prefix = out_dir + '/' + semi_qualified_prefix

        # Architecture configuration
        self.arch_specs = ArchSpecs(cfg['architecture'])
        self.arch_specs.generate_tables(
            cfg, semi_qualified_prefix, out_dir, self.out_prefix, log_level)

        # Problem configuration
        self.workload = Workload(cfg['problem'])
        self.model_logger.info('Problem configuration complete.')

        self.arch_props = ArchProperties(self.arch_specs)

        # Architecture constraints
        self.constraints = ArchConstraints(
            self.arch_props, self.workload, cfg['architecture_constraints'])
        self.model_logger.info('Architecture configuration complete.')

        # Mapping configuration
        self.mapping = Mapping(cfg['mapping'], self.arch_specs, self.workload)
        self.model_logger.info('Mapping construction complete.')

        # Validate mapping against architecture constraints
        if not self.constraints.satisfied_by(self.mapping):
            self.model_logger.error(
                'Mapping violates architecture constraints.')
            raise ValueError('Mapping violates architecture constraints.')

    def run(self):
        engine = Accelerator(self.arch_specs)

        eval_stat = engine.evaluate(
            self.mapping, self.workload, log_level=self.log_level)
        return eval_stat
