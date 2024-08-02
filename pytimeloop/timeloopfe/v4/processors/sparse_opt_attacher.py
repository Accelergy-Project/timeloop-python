"""Takes sparse optimizations from sparse optimizations lists and attaches them to the architecture.
"""
from ...common.processor import References2CopiesProcessor
from ...common.nodes import DictNode
from ...common.processor import Processor
from ...v4 import Specification


class SparseOptAttacherProcessor(Processor):
    """Takes sparse optimizations from sparse optimizations lists and attaches them to the architecture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, spec: Specification):
        super().process(spec)
        self.must_run_after(References2CopiesProcessor, spec)
        while spec.sparse_optimizations.targets:
            opt = spec.sparse_optimizations.targets.pop(0)
            nodes = spec.architecture.get_nodes_of_type(DictNode)
            for c in nodes:
                if c.get("name", None) == opt.target and "sparse_optimizations" in c:
                    c.combine_index("sparse_optimizations", opt)
                    break
            else:
                raise ValueError(
                    f"Sparse optimization target '{opt.target}' not found in "
                    f"the architecture. Problematic sparse optimization: {opt}"
                )
