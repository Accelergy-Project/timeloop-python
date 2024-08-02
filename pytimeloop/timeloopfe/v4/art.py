from ..common.nodes import DictNode, ListNode


class Art(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4")
        super().add_attr("tables", Tables)

    def isempty(self) -> bool:
        return self.tables.isempty()


class Tables(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Table)


class Table(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("area", float)


Art.declare_attrs()
Tables.declare_attrs()
Table.declare_attrs()
