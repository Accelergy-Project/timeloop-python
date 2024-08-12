from ..common.nodes import DictNode, ListNode


class Ert(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

        super().add_attr("version", default="0.4")
        super().add_attr("tables", Tables)

    def find_component(self, name: str):
        for table in self.tables:
            # Component names in ERT are compound, e.g.,
            # Parent1.Parent0.Component[0..NumOfComponent]
            last_component_in_compound_name: str = \
                table.name.split('[')[0].split('.')[-1]
            if name == last_component_in_compound_name:
                return table

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

    def find_action(self, name: str):
        for action in self.actions:
            if name == action.name:
                return action


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
