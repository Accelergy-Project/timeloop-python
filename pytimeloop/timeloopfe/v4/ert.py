import functools
from numbers import Number
import re

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
            # Parent1.Parent0[0..NumOfParent].Component[0..NumOfComponent]
            full_name = table.name
            full_name = re.sub('\[\d+\.\.\d+\]', '', full_name)
            last_component_in_compound_name: str = full_name.split('.')[-1]
            if name == last_component_in_compound_name:
                return table
        raise KeyError(f'Could not find component {name}')

    def isempty(self) -> bool:
        return self.tables.isempty()
    
    def to_dict(self):
        r = {}
        for t in self.tables:
            name = re.sub('\[\d+\.\.\d+\]', '', t.name).split('.')[-1]
            r[name] = {}
            for a in t.actions:
                r[(name, a.name)] = a.energy
        return r
            


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

    # @functools.cache
    def find_action(self, name: str):
        for action in self.actions:
            if name == action.name:
                return action
            
    def __hash__(self) -> int:
        return id(self)


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
        super().add_attr("energy", Number)


class ActionArguments(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)


for cls in [Ert, Tables, Table, Actions, Action, ActionArguments]:
    cls.declare_attrs()
