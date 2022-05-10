import bindings
from .mapping import Mapping
from .problem import Workload
from .model import ArchSpecs, SparseOptimizationInfo


class Accelerator(bindings.model.Accelerator):
    def __init__(self, arch_specs: ArchSpecs):
        super().__init__(arch_specs)

    def evaluate(self, mapping: Mapping, workload: Workload,
                 sparse_opts: SparseOptimizationInfo,
                 break_on_failure: bool = True):
        return super().evaluate(mapping, workload, sparse_opts,
                                break_on_failure)
