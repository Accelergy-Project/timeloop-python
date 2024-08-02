from ..common.nodes import DictNode
from .version import assert_version


class Mapspace(DictNode):
    """
    Top-level mapspace object

    Attributes:
        version (str): The version of the mapspace.
        template (str): The template to use for the mapspace. "ruby" for imperfect factorization, any other string for perfect factorization.

    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4", callfunc=assert_version)
        super().add_attr("template", str, "uber")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.template: str = self["template"]


Mapspace.declare_attrs()
