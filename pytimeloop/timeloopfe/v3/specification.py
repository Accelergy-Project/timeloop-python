from ..common.nodes import Node
from ..common.base_specification import BaseSpecification


class Specification(BaseSpecification):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("architecture", dict, {})
        super().add_attr("components", dict, {}, part_name_match=True)
        super().add_attr("constraints", dict, {})
        super().add_attr("mapping", list, [], part_name_match=True)
        super().add_attr("problem", dict, {})
        super().add_attr("sparse_optimizations", dict, {})
        super().add_attr("variables", dict, {})
        super().add_attr("mapspace", dict, {})
        super().add_attr("", part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        assert "_required_processors" not in kwargs, "Cannot set _required_processors"
        kwargs["_required_processors"] = []
        super().__init__(*args, **kwargs)

    @classmethod
    def from_yaml_files(cls, *args, **kwargs) -> "Specification":
        Node.reset_processor_elems()
        return super().from_yaml_files(*args, **kwargs)  # type: ignore


Specification.declare_attrs()
