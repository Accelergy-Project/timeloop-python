"""Takes constraints from constraints lists and attaches them to
objects in the architecture.
"""
from ...common.nodes import DictNode
from ...common.processor import Processor
from ...common.processor import References2CopiesProcessor
from ..specification import Specification


class ConstraintAttacherProcessor(Processor):
    """
    Takes constraints from constraints lists and attaches them to objects in the architecture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_target(self, x, spec: Specification):
        while x:
            constraint = x.pop(0)
            nodes = spec.architecture.get_nodes_of_type(DictNode)
            for c in nodes:
                if c.get("name", None) == constraint.target and "constraints" in c:
                    c["constraints"].combine_index(constraint.type, constraint)
                    break
            else:
                all_node_names = list(c.get("name") for c in nodes if "name" in c)
                raise ValueError(
                    f"Constraint target '{constraint.target}' not found in "
                    f"the architecture. Problematic constraint: {constraint}."
                    f"Available targets: {all_node_names}."
                )

    def process(self, spec: Specification):
        super().process(spec)
        self.must_run_after(References2CopiesProcessor, spec)
        self._process_target(spec.constraints.targets, spec)
        self._process_target(spec.mapping, spec)

    def declare_attrs(self, *args, **kwargs):
        return super().declare_attrs(*args, **kwargs)
