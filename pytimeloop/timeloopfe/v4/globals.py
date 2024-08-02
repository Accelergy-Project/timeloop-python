from ..common.nodes import CombinableListNode, DictNode
from .version import assert_version


class Globals(DictNode):
    """
    Top-level Globals key.

    Attributes:
        version (str): Version of the Timeloop file.
        environment_variables (EnvironmentVariables): Environment variables to be used.
        expression_custom_functions (ExpressionCustomFunctions): Paths to Python files containing functions to be used in expressions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4", callfunc=assert_version)
        super().add_attr("environment_variables", EnvironmentVariables, [])
        super().add_attr("expression_custom_functions", ExpressionCustomFunctions, [])
        super().add_attr("accelergy_plug_ins", AccelergyPlugIns, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.environment_variables: EnvironmentVariables = self["environment_variables"]
        self.expression_custom_functions: ExpressionCustomFunctions = self[
            "expression_custom_functions"
        ]
        self.accelergy_plug_ins: AccelergyPlugIns = self["accelergy_plug_ins"]


class EnvironmentVariables(DictNode):
    """
    Dictionary of environment variables.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        cls.recognize_all()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExpressionCustomFunctions(CombinableListNode):
    """
    A list of paths to Python files containing functions to be used in expressions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AccelergyPlugIns(CombinableListNode):
    """
    A list of paths to Python files containing Accelergy plug-ins.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


Globals.declare_attrs()
EnvironmentVariables.declare_attrs()
ExpressionCustomFunctions.declare_attrs()
