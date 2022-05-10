import logging
from pathlib import Path
import unittest

from bindings.model import BoundedAcceleratorPool, UnboundedAcceleratorPool

from pytimeloop.config import Config
from pytimeloop.engine import Accelerator
from pytimeloop.mapping import Mapping
from pytimeloop.model import ArchSpecs, SparseOptimizationInfo
from pytimeloop.problem import Workload

from .util import load_configs, TEST_TMP_DIR


class AcceleratorTest(unittest.TestCase):
    def setUp(self):
        CONFIG_DIR = Path('01-model-conv1d-2level')
        PATHS = ['arch/*.yaml',
                 'map/conv1d-2level-os.map.yaml',
                 'prob/*.yaml']
        TMP_DIR = TEST_TMP_DIR / 'AcceleratorTest'
        SEMI_QUAL_PREFIX = 'timeloop-model'
        OUT_PREFIX = TMP_DIR / SEMI_QUAL_PREFIX
        config = load_configs(CONFIG_DIR, PATHS)

        self.arch_specs = ArchSpecs(config['architecture'])
        self.arch_specs.generate_tables(
            config, str(SEMI_QUAL_PREFIX), str(TMP_DIR),
            str(OUT_PREFIX), logging.INFO)

        self.workload = Workload(config['problem'])

        self.mapping = Mapping(config['mapping'], self.arch_specs,
                               self.workload)

        self.sparse_opts = SparseOptimizationInfo(Config(),
                                                  self.arch_specs)

    def check_eval_stat(self, eval_stat, ref_area, ref_energy,
                        ref_cycles, ref_id=None):
        self.assertAlmostEqual(eval_stat.area, ref_area, 1)
        self.assertAlmostEqual(eval_stat.energy, ref_energy, 1)
        self.assertEqual(eval_stat.cycles, ref_cycles)
        if ref_id is not None:
            self.assertEqual(eval_stat.id, ref_id)

    def test_accelerator(self):
        acc = Accelerator(self.arch_specs)

        eval_stat = acc.evaluate(self.mapping, self.workload,
                                 self.sparse_opts)
        self.check_eval_stat(eval_stat, 748186.1, 340.7, 48)

    def test_unbounded_accelerator_pool(self):
        NTHREADS = 4
        acc = UnboundedAcceleratorPool(self.arch_specs, NTHREADS)

        eval_id = acc.evaluate(self.mapping, self.workload,
                               self.sparse_opts, True)
        eval_stat = acc.get_result()

        self.check_eval_stat(eval_stat, 748186.1, 340.7, 48, eval_id)

    def test_bounded_accelerator_pool(self):
        NTHREADS = 4
        acc = BoundedAcceleratorPool(self.arch_specs, 2*NTHREADS,
                                     NTHREADS)

        eval_id = acc.evaluate(self.mapping, self.workload,
                               self.sparse_opts, True)
        eval_stat = acc.get_result()

        self.check_eval_stat(eval_stat, 748186.1, 340.7, 48, eval_id)
