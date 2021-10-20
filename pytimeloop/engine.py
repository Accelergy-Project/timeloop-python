from bindings import get_problem_shape, NativeEngine, NativeAcceleratorPool
from .model import ArchSpecs, SparseOptimizationInfo
from .problem import Workload
from .mapping import Mapping

import logging


class Accelerator(NativeEngine):
    def __init__(self, specs: ArchSpecs):
        super().__init__(specs)
        self.specs = specs
        self.pre_eval_status = None
        self.eval_status = None

    def evaluate(self, mapping: Mapping, workload: Workload,
                 sparse_optimizations: SparseOptimizationInfo,
                 break_on_failure=False, auto_bypass_on_failure=True,
                 return_stats=True, log_level=logging.INFO):
        """
        Evaluate a given `Mapping` and `Workload`.

        Args:
            mapping: an instance of `Mapping` to evaluate.
            workload: an instance of `Workload` to evaluate.
            sparse_optimizations: TODO
            break_on_failure: TODO
            auto_bypass_on_failure: if True, performs pre-evaluation 
                check and bypasses architecture levels that couldn't 
                be mapped. `mapping` might be modified if this is 
                turned on.
            log_level: the level to perform logging. Levels can be
                found in the documentation for thestandard logging
                library.

        Returns:
            AcceleratorEvalResult: result of accelerator evaluation. 
                Has properties `eval_stat`, `utilization`, 
                `total_energy`, `total_maccs`.
        """
        # Setup logger
        logger = logging.getLogger(__name__ + '.' + __class__.__name__)
        logger.setLevel(log_level)

        # Perform pre-evaluation check and auto-bypass
        level_names = self.specs.level_names()
        self.pre_eval_status = self.pre_evaluation_check(
            mapping, workload, sparse_optimizations, False)
        if auto_bypass_on_failure:
            for level, status in enumerate(self.pre_eval_status):
                if not status.success:
                    logger.warning("Couldn't map level ", level_names[level],
                                   ': ', self.pre_eval_status[level].fail_reason,
                                   ', auto-bypassing.')
            if not status.success:
                for pvi in range(get_problem_shape().num_data_spaces):
                    mapping.datatype_bypass_nest[pvi].reset(level-1)
        else:
            return AcceleratorEvalResult(eval_status=None,
                                         pre_eval_status=self.pre_eval_status)

        # Perform evaluation
        eval_status = super().evaluate(mapping, workload, sparse_optimizations)
        self.eval_status = eval_status
        for level, status in enumerate(eval_status):
            if not status.success:
                logger.error("Coulnd't map level ", level_names[level], ': ',
                             self.pre_eval_status[level].fail_reason)
                return AcceleratorEvalResult(eval_status, self.pre_eval_status)

        if self.is_evaluated():
            logger.info(
                'Utilization = {}\n'.format(self.utilization()) +
                'pJ/Algorithmic-Compute = {}\n'.format(
                    self.energy() / self.get_topology().algorithmic_computes()) +
                'pJ/Compute = {}'.format(
                    self.energy() / self.get_topology().actual_computes())
            )

        if return_stats:
            return self.get_stats(mapping)
        else:
            return AcceleratorEvalResult(eval_status, self.pre_eval_status)

    def get_stats(self, mapping):
        topology = self.get_topology()
        pretty_printed_mapping = mapping.pretty_print(
            self.specs.storage_level_names(),
            topology.utilized_capacities(),
            topology.tile_sizes())
        return AcceleratorEvalResult(self.eval_status, self.pre_eval_status,
                                     self.utilization(), self.energy(),
                                     self.area(), self.cycles(),
                                     topology.algorithmic_computes(),
                                     topology.actual_computes(),
                                     topology.last_level_accesses(),
                                     topology.tile_sizes(),
                                     self.pretty_print_stats(),
                                     pretty_printed_mapping)


class AcceleratorPool(NativeAcceleratorPool):
    """A pool of accelerators."""
    def __init__(self, specs: ArchSpecs, num_workers: int):
        super().__init__(specs, num_workers)

    def evaluate(self, mapping: Mapping, workload: Workload,
                 sparse_opts: SparseOptimizationInfo, break_on_failure=False,
                 auto_bypass_on_failure=True):
        return super().evaluate(mapping, workload, sparse_opts,
                                break_on_failure, auto_bypass_on_failure)

    def get_result(self):
        res = super().get_result()
        return AcceleratorEvalResult(res.eval_status, res.pre_eval_status,
                                     res.utilization, res.energy, res.area,
                                     res.cycles, res.algorithmic_computes,
                                     res.actual_computes,
                                     res.last_level_accesses,
                                     task_id=res.id)


class AcceleratorEvalResult:
    """
    Evaluation result from calling `Accelerator.evaluate`.

    Attributes:
        eval_status: evaluation status returned by Timeloop method
            `Engine.Evaluate`.
        pre_eval_status: pre-evaluation status returned by Timeloop
            method `Engine.PreEvaluationCheck`.
        utilization: utilization of accelerator for the workload and
            mapping.
        energy: total energy consumed by the accelerator for the
            workload and mapping.
        area: TODO
        algorithmic_compute: TODO
        actual_compute: TODO
        tile_sizes: TODO
    """

    def __init__(self, eval_status, pre_eval_status, utilization=None,
                 energy=None, area=None, cycles=None, algorithmic_compute=None,
                 actual_compute=None, last_level_accesses = None,
                 tile_sizes=None, pretty_printed_stats='',
                 pretty_printed_mapping='', task_id=None):
        self.eval_status = eval_status
        self.pre_eval_status = pre_eval_status
        self.utilization = utilization
        self.energy = energy
        self.area = area
        self.cycles = cycles
        self.algorithmic_compute = algorithmic_compute
        self.actual_compute = actual_compute
        self.last_level_accesses = last_level_accesses
        self.tile_sizes = tile_sizes
        self.pretty_printed_stats = pretty_printed_stats
        self.pretty_printed_mapping = pretty_printed_mapping
        self.task_id = task_id

    def pretty_print_stats(self):
        return self.pretty_printed_stats

    def pretty_print_mapping(self):
        return self.pretty_printed_mapping

    def __str__(self):
        res = 'Eval status: ' + self.eval_status.__str__() + '\n'
        res += 'Pre-eval status: ' + self.pre_eval_status.__str__() + '\n'
        res += 'Mapping:\n'
        res += self.pretty_printed_mapping
        res += 'Evaluation stats:\n'
        res += self.pretty_printed_stats
        return res
