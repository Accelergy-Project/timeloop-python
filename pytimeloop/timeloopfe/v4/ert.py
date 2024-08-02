from ..common.nodes import DictNode, ListNode


class Ert(DictNode):
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
        super().add_attr("actions", Actions)


class Actions(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

        super().add_attr("", Action)


class Action(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

        super().add_attr("name", str)
        super().add_attr("arguments", ActionArguments)
        super().add_attr("energy", float)


class ActionArguments(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)


for cls in [Ert, Tables, Table, Actions, Action, ActionArguments]:
    cls.declare_attrs()
