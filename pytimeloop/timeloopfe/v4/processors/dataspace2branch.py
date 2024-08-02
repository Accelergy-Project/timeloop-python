"""Resolves which data spaces are kept in which branches."""
from typing import Set
from .constraint_attacher import ConstraintAttacherProcessor
from .constraint_macro import ConstraintMacroProcessor
from ..arch import Branch, Parallel
from ...common.nodes import Node
from ..constraints import Dataspace
from ...common.processor import Processor
from ...common.processor import References2CopiesProcessor
from ...v4 import Specification


class Dataspace2BranchProcessor(Processor):
    """Resolves which data spaces are kept in which branches."""

    def get_problem_ds_names(self, spec) -> Set[str]:
        return set([x.name for x in spec.problem.shape.data_spaces])

    def _get_kept_dataspaces(self, b: Node) -> Set[str]:
        return set().union(*[d.keep for d in b.get_nodes_of_type(Dataspace)])

    def _parse_branch(
        self, branch: Branch, dataspaces: Set[str], spec: Specification
    ) -> Set[str]:
        subnodes = branch.nodes
        all_ds = self.get_problem_ds_names(spec)
        if isinstance(branch, Parallel):
            data_spaces_remaining = set(dataspaces)
            idx2keep = [self._get_kept_dataspaces(s) for s in subnodes]
            for i, s1 in enumerate(subnodes):
                for j, s2 in enumerate(subnodes[i + 1 :]):
                    shared = idx2keep[i].intersection(idx2keep[j + i + 1])
                    if shared:
                        raise ValueError(
                            f"DataSpaces {shared} are kept in two peer "
                            f"branches {s1} and {s2}. Each data space can "
                            f"only be kept in one branch. Full !Parallel node: "
                            f"{branch}."
                        )

            remaining_ds = data_spaces_remaining - set().union(*idx2keep)
            if remaining_ds:
                ds_list = "[" + ", ".join(remaining_ds) + "]"
                raise ValueError(
                    f"Can not find branch for {remaining_ds} in "
                    f"{branch}. If you would like to bypass all branches, add "
                    f"a new branch '- !Container "
                    f"{{constraints: {{dataspaces: {{keep: {ds_list}}}}}}}'"
                    f"to the !Parallel node. If you would like a data space to "
                    f"be kept in one branch, add a keep constraint to something "
                    f"in that branch."
                )

            for i, s in enumerate(subnodes):
                keep = idx2keep[i]
                bypass = all_ds - keep
                self.logger.info(
                    'Branch "%s" keeps %s and bypasses %s.', s, keep, bypass
                )
                for ds in s.get_nodes_of_type(Dataspace):
                    ds.combine(Dataspace(bypass=list(bypass)))
                if isinstance(s, Branch):
                    self._parse_branch(s, keep, spec)
        else:
            for s in subnodes:
                if isinstance(s, Branch):
                    self._parse_branch(s, dataspaces, spec)

    def process(self, spec: Specification):
        super().process(spec)
        self.must_run_after(References2CopiesProcessor, spec)
        self.must_run_after(ConstraintMacroProcessor, spec, ok_if_not_found=True)
        self.must_run_after(ConstraintAttacherProcessor, spec)
        self._parse_branch(spec.architecture, self.get_problem_ds_names(spec), spec)
