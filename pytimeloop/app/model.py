from bindings import ArchProperties
from pytimeloop.config import Config
from pytimeloop.engine import Accelerator
from pytimeloop.model import ArchSpecs
from pytimeloop.mapping import ArchConstraints, Mapping
from pytimeloop.problem import Workload

import sys
import logging


class Model:
    def __init__(self, cfg: Config, out_dir: str, verbose=False,
                 auto_bypass_on_failure=False, out_prefix='',
                 log_level=logging.INFO):
        # Setup logger
        self.logger = logging.getLogger('pytimeloop.app.Model')
        formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        # timeloop-model configurations
        self.verbose = verbose
        self.auto_bypass_on_failure = auto_bypass_on_failure
        self.out_prefix = out_prefix
        semi_qualified_prefix = 'timeloop-model'
        self.out_prefix = out_dir + '/' + semi_qualified_prefix

        # Architecture configuration
        self.arch_specs = ArchSpecs(cfg['architecture'])
        self.arch_specs.generate_tables(
            cfg, semi_qualified_prefix, out_dir, self.out_prefix)

        # Problem configuration
        self.workload = Workload(cfg['problem'])
        self.logger.info('Problem configuration complete.')

        self.arch_props = ArchProperties(self.arch_specs)

        # Architecture constraints
        self.constraints = ArchConstraints(
            self.arch_props, self.workload, cfg['architecture_constraints'])
        self.logger.info('Architecture configuration complete.')

        # Mapping configuration
        self.mapping = Mapping(cfg['mapping'], self.arch_specs, self.workload)
        self.logger.info('Mapping construction complete.')

        # Validate mapping against architecture constraints
        if not self.constraints.satisfied_by(self.mapping):
            self.logger.error('Mapping violates architecture constraints.')
            raise ValueError('Mapping violates architecture constraints.')

    def run(self):
        stats_fname = self.out_prefix + 'stats.txt'
        xml_fname = self.out_prefix + '.map+stats.xml'
        map_txt_fname = self.out_prefix + '.map.txt'

        engine = Accelerator(self.arch_specs)

        eval_stat = engine.evaluate(
            self.mapping, self.workload, False, True, True)
        return eval_stat
