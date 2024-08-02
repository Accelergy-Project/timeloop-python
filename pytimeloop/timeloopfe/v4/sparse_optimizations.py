from typing import List, Optional

from ..common.nodes import DictNode, ListNode, isempty, CombinableListNode
from .version import assert_version


class SparseOptimizations(DictNode):
    """
    Top-level class for sparse optimizations.

    Attributes:
        version (str): The version of the sparse optimizations.
        targets (SparseOptimizationsList): A list of sparse optimizations.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4", callfunc=assert_version)
        super().add_attr("targets", SparseOptimizationsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.targets: SparseOptimizationsList = self["targets"]


class SparseOptimizationsList(ListNode):
    """
    A list of sparse optimizations."""

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", SparseOptimizationGroup)


class SparseOptimizationGroup(DictNode):
    """
    A group of sparse optimizations.

    Attributes:
        target (str): The target of the sparse optimization group.
        action_optimization (ActionOptimization): The action optimization associated with the group.
        representation_format (RepresentationFormat): The representation format associated with the group.
        compute_optimization (ComputeOptimization): The compute optimization associated with the group.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("target", str, None)
        super().add_attr("action_optimization", ActionOptimizationList, [])
        super().add_attr("representation_format", RepresentationFormat, {})
        super().add_attr("compute_optimization", ComputeOptimizationList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target: str = self["target"]
        self.action_optimization: ActionOptimization = self["action_optimization"]
        self.representation_format: RepresentationFormat = self["representation_format"]
        self.compute_optimization: ComputeOptimization = self["compute_optimization"]

    def isempty(self) -> bool:
        """
        Check if the sparse optimization group is empty.

        Returns:
            bool: True if the group is empty, False otherwise.
        """
        return (
            isempty(self.get("action_optimization", None))
            and isempty(self.get("representation_format", None))
            and isempty(self.get("compute_optimization", None))
        )


class RepresentationFormat(DictNode):
    """
    A representation format sparse optimization.

    Attributes:
        data_spaces (RepresentationProblemDataspaceList): A list of data spaces.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("data_spaces", RepresentationProblemDataspaceList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_spaces: RepresentationProblemDataspaceList = self["data_spaces"]


class RepresentationProblemDataspaceList(ListNode):
    """
    A list of representation problem dataspaces.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", RepresentationDataSpace)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RepresentationDataSpace(DictNode):
    """
    Contains the representation format for a data space.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("ranks", RepresentationRankList)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target: str = self["name"]
        self.ranks: RepresentationRankList = self["ranks"]


class RepresentationRankList(ListNode):
    """
    A list of ranks to be used in the representation format.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", RepresentationRank)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranks: List[RepresentationRank] = self


class RepresentationRank(DictNode):
    """
    A representation rank.

    Attributes:
        name (str): The name of the rank.
        format (str): The format of the rank. One of "CP", "B", "RLE", "UOP".
        metadata_word_bits (int): The number of metadata word bits.
        payload_word_bits (int): The number of payload word bits.
        flattened_rankIDs (list): A list of flattened rank IDs.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("format", ("CP", "B", "RLE", "UOP"))
        super().add_attr("metadata_word_bits", int, None)
        super().add_attr("payload_word_bits", int, None)
        super().add_attr("flattened_rankIDs", (list), None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.format: str = self["format"]
        self.metadata_word_bits: Optional[int] = self["metadata_word_bits"]
        self.flattened_rankIDs: list = self["flattened_rankIDs"]


class ActionOptimizationList(ListNode):
    """
    A list of action optimizations.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", ActionOptimization)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ComputeOptimizationList(ListNode):
    """
    A list of compute optimizations.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", ComputeOptimization)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ComputeOptimization(DictNode):
    """
    A compute optimization.

    Attributes:
        type (str): The type of compute optimization. One of "gating", "skipping".
        options (ActionOptimizationOptionList): A list of action optimization options.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("type", ComputeOptimizationTypeList)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ComputeOptimizationTypeList(CombinableListNode):
    """
    A list of compute optimizations.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", ("gating", "skipping"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ActionOptimization(DictNode):
    """
    An action optimization.

    Attributes:
        type (str): The type of action optimization. One of "gating", "skipping", "spatial_skipping".
        options (ActionOptimizationOptionList): A list of action optimization options.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("type", ("gating", "skipping", "spatial_skipping"))
        super().add_attr("options", ActionOptimizationOptionList)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = self["type"]


class ActionOptimizationOptionList(ListNode):
    """
    A list of action optimization options.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", ActionOptimizationOption)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ActionOptimizationOption(DictNode):
    """
    An action optimization option.

    Attributes:
        target (str): The target of the optimization.
        condition_on (list): Which tensor(s) to condition on.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("target", str)
        super().add_attr("condition_on", list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target: str = self["target"]
        self.condition_on: list = self["condition_on"]


SparseOptimizationGroup.declare_attrs()
RepresentationRankList.declare_attrs()
RepresentationProblemDataspaceList.declare_attrs()
RepresentationFormat.declare_attrs()
RepresentationDataSpace.declare_attrs()
RepresentationRank.declare_attrs()
ActionOptimizationList.declare_attrs()
ActionOptimization.declare_attrs()
ActionOptimizationOptionList.declare_attrs()
ActionOptimizationOption.declare_attrs()
SparseOptimizations.declare_attrs()
SparseOptimizationsList.declare_attrs()
ComputeOptimization.declare_attrs()
