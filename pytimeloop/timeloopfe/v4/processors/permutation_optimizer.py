"""Optimizes permutation by pruning superfluous permutations."""
from .constraint_attacher import (
    ConstraintAttacherProcessor,
)
from .constraint_macro import ConstraintMacroProcessor
from .dataspace2branch import Dataspace2BranchProcessor
from ..arch import Leaf, Storage
from ...common.processor import Processor
from ...common.processor import References2CopiesProcessor
from ...v4 import Specification


class PermutationOptimizerProcessor(Processor):
    """Optimizes permutation by pruning superfluous permutations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, spec: Specification):
        super().process(spec)
        # Assert that the constraint attacher processor has already run
        self.must_run_after(ConstraintAttacherProcessor, spec)
        self.must_run_after(References2CopiesProcessor, spec)
        self.must_run_after(ConstraintMacroProcessor, spec, ok_if_not_found=True)
        self.must_run_after(Dataspace2BranchProcessor, spec, ok_if_not_found=True)
        problem = spec.problem

        constraints = []
        for c in spec.get_nodes_of_type(Leaf):
            if isinstance(c, Storage):
                constraints.append(c.constraints.temporal)
            if c.spatial.get_fanout() > 1:
                constraints.append(c.constraints.spatial)

        for c in constraints:
            for d, _, factor in c.factors.get_split_factors():
                if int(factor) == 1 and d not in c.permutation:
                    c.permutation.insert(0, d)
            for d in problem.shape.dimensions:
                if problem.instance[d] == 1 and d not in c.permutation:
                    c.permutation.insert(0, d)
