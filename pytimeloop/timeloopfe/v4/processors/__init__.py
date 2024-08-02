"""Standard suite of processors for timeloopfe."""

from . import (
    constraint_attacher,
    constraint_macro,
    enable_dummy_table,
    permutation_optimizer,
    sparse_opt_attacher,
    required_actions,
    dataspace2branch,
)
from ...common.processor import Processor, References2CopiesProcessor

ConstraintAttacherProcessor = constraint_attacher.ConstraintAttacherProcessor
ConstraintMacroProcessor = constraint_macro.ConstraintMacroProcessor
Dataspace2BranchProcessor = dataspace2branch.Dataspace2BranchProcessor
EnableDummyTableProcessor = enable_dummy_table.EnableDummyTableProcessor
# MathProcessor = math.MathProcessor
PermutationOptimizerProcessor = permutation_optimizer.PermutationOptimizerProcessor
SparseOptAttacherProcessor = sparse_opt_attacher.SparseOptAttacherProcessor
RequiredActionsProcessor = required_actions.RequiredActionsProcessor
# Order matters here. The processors will be run in the order they appear in
# this list.


REQUIRED_PROCESSORS = [
    ConstraintAttacherProcessor,
    SparseOptAttacherProcessor,
    ConstraintMacroProcessor,
    Dataspace2BranchProcessor,
    PermutationOptimizerProcessor,
    RequiredActionsProcessor,
]
