from collections.abc import Mapping, Sequence

from ..common.nodes import DictNode, ListNode


class FusedProblem(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Einsum)


class Einsum(DictNode):
    pass


def cast_to_problem_or_fused_problem(value):
    """
    Casts `value` to `Problem` if value contains one Einsum; otherwise,
    casts `value` to `FusedProblem`.
    
    This function is for use as the `callfunc` argument to `add_attr`
    in v4.Specification.
    """
    if isinstance(value, Mapping):
        return Problem(**value)
    elif isinstance(value, Sequence):
        return FusedProblem(value)
    else:
        raise RuntimeError('Type is wrong')
