from bindings import get_problem_shape, NativeEngine
from .model import ArchSpecs
from .problem import Workload
from .mapping import Mapping


class Accelerator(NativeEngine):
    def __init__(self, specs: ArchSpecs):
        super().__init__(specs)
        self.specs = specs

    def evaluate(self, mapping: Mapping, workload: Workload,
                 break_on_failure=False, auto_bypass_on_failure=True,
                 verbose=False):
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
            verbose: if True, auto-bypass and evaluation messages 
                will be printed.

        Returns:
            AcceleratorEvalStat: result of accelerator evaluation. 
                Has properties `eval_stat`, `utilization`, 
                `total_energy`, `total_maccs`.

        Todo:
            Capture pretty print results somehow.
        """
        level_names = self.specs.level_names()
        if auto_bypass_on_failure:
            pre_eval_stat = self.pre_evaluation_check(mapping, workload, False)
            for level, status in enumerate(pre_eval_stat):
                if not status.success and self.verbose:
                    print("ERROR: couldn't map level ", level_names[level],
                          ': ', pre_eval_stat[level].fail_reason,
                          ', auto-bypassing.')
            if not status.success:
                for pvi in range(get_problem_shape().num_data_spaces):
                    mapping.datatype_bypass_nest[pvi].reset(level-1)

        eval_stat = super().evaluate(mapping, workload)
        for level, status in enumerate(eval_stat):
            if not status.success:
                print("ERROR: coulnd't map level ", level_names[level], ': ',
                      pre_eval_stat[level].fail_reason)
                return None

        if self.is_evaluated():
            if verbose:
                print('Utilization = ', self.utilization(), ' | pJ/MACC',
                      self.energy() / self.get_topology().maccs())

        return AcceleratorEvalStat(eval_stat, pre_eval_stat, self.utilization(),
                                   self.energy(), self.get_topology().maccs())


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

    def __init__(self, eval_stat, pre_eval_stat, utilization, total_energy,
                 total_maccs):
        self.eval_stat = eval_stat
        self.pre_eval_stat = pre_eval_stat
        self.utilization = utilization
        self.total_energy = total_energy
        self.total_maccs = total_maccs
