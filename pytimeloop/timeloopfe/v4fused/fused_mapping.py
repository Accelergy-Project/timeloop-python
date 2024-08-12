from ..common.nodes import DictNode, ListNode


class FusedMapping(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

        super().add_attr("type", default="fused")
        super().add_attr("nodes", FusedMappingNodes)


class FusedMappingNodes(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

        super().add_attr("", FusedMappingNode)


class FusedMappingNode(DictNode):
    pass


def cast_to_constraint_list_or_fused_mapping(value):
    """
    Casts nodes to `FusedMapping` if key `type` exists and the value
    is 'fused'. This function is for use as the `callfunc` argument
    to `add_attr` in v4.Specification.
    """
    if 'type' in value and value['type'] == 'fused':
        return FusedMapping(**value)
    else:
        return ConstraintsList(**value)
