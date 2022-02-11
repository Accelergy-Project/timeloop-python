import bindings
from bindings import get_problem_shape, UnboundedAcceleratorPool,\
                     BoundedAcceleratorPool
from .model import ArchSpecs, SparseOptimizationInfo
from .problem import Workload
from .mapping import Mapping

class Accelerator(bindings.Accelerator):
    def __init__(self, arch_specs: ArchSpecs):
        super().__init__(arch_specs)

    def evaluate(self, mapping: Mapping, workload: Workload,
                 sparse_opts: SparseOptimizationInfo,
                 break_on_failure: bool = True):
        return super().evaluate(mapping, workload, sparse_opts,
                                break_on_failure)
