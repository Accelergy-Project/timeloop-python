from bindings import get_problem_shape, NativeEngine
from .model import ArchSpecs
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
                 break_on_failure=False, auto_bypass_on_failure=True,
                 log_level=logging.INFO):
        """
        Evaluate a given `Mapping` and `Workload`.

        Args:
            mapping: an instance of `Mapping` to evaluate.
            workload: an instance of `Workload` to evaluate.
            break_on_failure: 
            auto_bypass_on_failure: if True, performs pre-evaluation 
                check and bypasses architecture levels that couldn't 
                be mapped. `mapping` might be modified if this is 
                turned on.
            log_level: the level to perform logging. Levels can be
                found in the documentation for thestandard logging
                library.

        Returns:
            AcceleratorEvalStat: result of accelerator evaluation. 
                Has properties `eval_stat`, `utilization`, 
                `total_energy`, `total_maccs`.

        Todo:
            Capture pretty print results somehow.
        """
        # Setup logger
        logger = logging.getLogger(__name__ + '.' + __class__.__name__)
        logger.setLevel(log_level)

        level_names = self.specs.level_names()
        if auto_bypass_on_failure:
            pre_eval_status = self.pre_evaluation_check(
                mapping, workload, False)
            self.pre_eval_status = pre_eval_status
            for level, status in enumerate(pre_eval_status):
                if not status.success:
                    logger.warning("Couldn't map level ", level_names[level],
                                   ': ', self.pre_eval_status[level].fail_reason,
                                   ', auto-bypassing.')
            if not status.success:
                for pvi in range(get_problem_shape().num_data_spaces):
                    mapping.datatype_bypass_nest[pvi].reset(level-1)

        eval_status = super().evaluate(mapping, workload)
        self.eval_status = eval_status
        for level, status in enumerate(eval_status):
            if not status.success:
                logger.error("Coulnd't map level ", level_names[level], ': ',
                             self.pre_eval_status[level].fail_reason)
                return None

        if self.is_evaluated():
            logger.info(
                'Utilization = {} | pJ/MAC = {}'.format(
                    self.utilization(),
                    self.energy() / self.get_topology().maccs()
                )
            )

        return AcceleratorEvalStat(eval_status, pre_eval_status,
                                   self.utilization(), self.energy(),
                                   self.get_topology().maccs())

    def get_stats(self):
        return AcceleratorEvalStat(self.eval_status, self.pre_eval_status,
                                   self.utilization(), self.energy(),
                                   self.get_topology().maccs())


class AcceleratorEvalStat:
    """
    Evaluation result from calling `Accelerator.evaluate`.

    Attributes:
        eval_stat: evaluation status returned by Timeloop method
            `Engine.Evaluate`.
        pre_eval_stat: pre-evaluation status returned by Timeloop
            method `Engine.PreEvaluationCheck`.
        utilization: utilization of accelerator for the workload and
            mapping.
        total_energy: total energy consumed by the accelerator for the
            workload and mapping.
        total_macc: total number of MACC operations performed for the
            workload and mapping.
    """

    def __init__(self, eval_status, pre_eval_status, utilization, total_energy,
                 total_maccs):
        self.eval_status = eval_status
        self.pre_eval_status = pre_eval_status
        self.utilization = utilization
        self.total_energy = total_energy
        self.total_maccs = total_maccs
