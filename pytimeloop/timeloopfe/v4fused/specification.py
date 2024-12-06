from ..common.nodes import ListNode
from ..v4.arch import Architecture
from ..v4.art import Art
from ..v4.ert import Ert
from ..v4.variables import Variables
from ..v4.components import Components
from ..v4.globals import Globals
from .fused_mapping import FusedMapping, cast_to_constraint_list_or_fused_mapping
from .fused_problem import FusedProblem, cast_to_problem_or_fused_problem
from ..common.processor import ProcessorError, References2CopiesProcessor

from typing import Any, Dict, List, Optional, Union
from ..common.base_specification import BaseSpecification, class2obj


class Specification(BaseSpecification):
    """
    A top-level class for the Timeloop specification.

    Attributes:
        architecture: The top-level architecture description.
        components: List of compound components.
        constraints: Additional constraints on the architecture and mapping.
        mapping: Additional constraints on the architecture and mapping.
        problem: The problem specification.
        variables: Variables to be used in parsing.
        mapper: Directives to control the mapping process.
        sparse_optimizations: Additional sparse optimizations available to the architecture.
        mapspace: The top-level mapspace description.
        globals: Global inclusion of extra parsing functions and environment variables.

    Note: Inherits from BaseSpecification.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("architecture", Architecture)
        super().add_attr(
            "components", Components, {"version": 0.4}, part_name_match=True
        )
        super().add_attr(
            "mapping",
            required_type=FusedMapping,
            default=[],
            part_name_match=True
        )
        super().add_attr(
            "problem",
            required_type=FusedProblem,
            default=[]
        )
        super().add_attr("mapping_constraints", default={})
        super().add_attr("variables", Variables, {"version": 0.4})
        super().add_attr("globals", Globals, {"version": 0.4}, part_name_match=True)
        super().add_attr("ERT", Ert, {"version": 0.4, "tables": []})
        super().add_attr("ART", Art, {"version": 0.4, "tables": []})

    def __init__(self, *args, **kwargs):
        kwargs["_required_processors"] = []
        super().__init__(*args, **kwargs)
        self.architecture = self["architecture"]
        self.mapping = self["mapping"]
        self.problem = self["problem"]
        self.variables = self["variables"]
        self.components: ListNode = self["components"]

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        if self.needs_processing([References2CopiesProcessor]):
            raise ProcessorError(
                f"Must run References2CopiesProcessor before "
                f"parsing expressions. Call process() with "
                f"any arguments."
            )
        for p in self.processors:
            if self.needs_processing([p], pre_parse=True):
                class2obj(p).pre_parse_process(self)
                self._processors_run_pre_parse.append(p)

        symbol_table = {} if symbol_table is None else symbol_table.copy()
        parsed_ids = set() if parsed_ids is None else parsed_ids
        parsed_ids.add(id(self))
        parsed_ids.add(id(self.variables))
        symbol_table["spec"] = self
        parsed_variables = self.variables.parse_expressions(symbol_table, parsed_ids)
        symbol_table.update(parsed_variables)
        symbol_table["variables"] = parsed_variables
        super().parse_expressions(symbol_table, parsed_ids)

    def to_diagram(
        self,
        container_names: Union[str, List[str]] = (),
        ignore_containers: Union[str, List[str]] = (),
    ) -> "pydot.Graph":
        from ..v4.processors.to_diagram_processor import ToDiagramProcessor

        s = self._process()
        proc = ToDiagramProcessor(container_names, ignore_containers, spec=s)
        return proc.process(s)

    # def _parse_timeloop_output(
    #     self, timeloop_output_dir: str, prefix: str
    # ) -> OutputStats:
    #     return parse_timeloop_output(self, timeloop_output_dir, prefix)


Specification.declare_attrs()
