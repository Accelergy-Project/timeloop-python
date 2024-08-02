from numbers import Number
from typing import Any, Dict, Optional, Union
from ..common.nodes import DictNode, CombinableListNode, ListNode, FlatteningListNode
from .version import assert_version


class Components(DictNode):
    """
    A collection of components.

    Attributes:
        version (str): The version of the components.
        classes (FlatteningListNode): The list of classes associated with the components.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4", callfunc=assert_version)
        super().add_attr("classes", FlatteningListNode, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.classes: FlatteningListNode = self["classes"]

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        pass


class ComponentsList(FlatteningListNode):
    """
    A list of components.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, *kwargs)
        super().add_attr("", CompoundComponent)


class CompoundComponent(DictNode):
    """
    Represents a compound component.

    Attributes:
        name (str): The name of the compound component.
        attributes (ComponentAttributes): The attributes of the compound component.
        subcomponents (SubcomponentList): The list of subcomponents.
        actions (ActionsList): The list of actions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("attributes", ComponentAttributes, {})
        super().add_attr("subcomponents", SubcomponentList, [])
        super().add_attr("actions", ActionsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.attributes: ComponentAttributes = self["attributes"]
        self.subcomponents: SubcomponentList = self["SubcomponentList"]
        self.actions: ActionsList = self["actions"]


class SubcomponentList(ListNode):
    """
    A list of subcomponents.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Subcomponent, part_name_match=True, no_change_key=True)


class Subcomponent(DictNode):
    """
    A subcomponent.

    Attributes:
        name (str): The name of the subcomponent.
        attributes (ComponentAttributes): The attributes of the subcomponent.
        area_scale (Union[Number, str]): The area share of the subcomponent.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("attributes", ComponentAttributes, {})
        super().add_attr("area_scale", (Number, str), 1)
        super().add_attr("energy_scale", (Number, str), 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.attributes: ComponentAttributes = self["attributes"]
        self.area_scale: Union[Number, str] = self["area_scale"]
        self.energy_scale: Union[Number, str] = self["energy_scale"]


class ComponentAttributes(DictNode):
    """
    Represents the attributes of a component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", part_name_match=True, no_change_key=True)


class ActionsList(ListNode):
    """
    A list of actions for a component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Action, part_name_match=True, no_change_key=True)


class Action(DictNode):
    """
    Represents an action that a component can perform.

    Attributes:
        name (str): The name of the action.
        arguments (DictNode): The arguments of the action.
        subcomponents (ActionSubcomponentsList): The subcomponents of the action.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("arguments", DictNode, {})
        super().add_attr("subcomponents", ActionSubcomponentsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.arguments: DictNode = self["arguments"]
        self.subcomponents: ActionSubcomponentsList = self["subcomponents"]


class ActionSubcomponentsList(ListNode):
    """
    A list of actions that may be taken as subactions of an action.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(
            "",
            SubcomponentActionGroup,
            part_name_match=True,
            no_change_key=True,
        )


class SubcomponentActionGroup(DictNode):
    """
    A group of subactions taken by a particular subcomponent.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("actions", SubcomponentActionList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.actions: SubcomponentActionList = self["actions"]


class SubcomponentActionList(ListNode):
    """
    A list of subcomponent actions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", SubcomponentAction)


class SubcomponentAction(DictNode):
    """
    A subcomponent action.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("arguments", DictNode, {})
        super().add_attr("energy_scale", (str, float), 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.arguments: DictNode = self["arguments"]
        self.energy_scale: Union[str, float] = self["energy_scale"]


Components.declare_attrs()
ComponentsList.declare_attrs()
CompoundComponent.declare_attrs()
SubcomponentList.declare_attrs()
Subcomponent.declare_attrs()
ComponentAttributes.declare_attrs()
ActionsList.declare_attrs()
Action.declare_attrs()
ActionSubcomponentsList.declare_attrs()
SubcomponentActionGroup.declare_attrs()
SubcomponentActionList.declare_attrs()
SubcomponentAction.declare_attrs()
