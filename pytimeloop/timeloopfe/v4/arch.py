from abc import ABC
from logging import Logger
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union
from ..common.nodes import DictNode, ListNode, Node
from . import constraints
from .sparse_optimizations import SparseOptimizationGroup
from .version import assert_version

BUFFER_CLASSES = ("DRAM", "SRAM", "regfile", "smartbuffer", "storage")
COMPUTE_CLASSES = ("mac", "intmac", "fpmac", "compute")
NETWORK_CLASSES = ("XY_NoC", "Legacy", "ReductionTree", "SimpleMulticast")
NOTHING_CLASSES = ("nothing",)


class ArchNode(Node):
    """
    A node in the architecture hierarchy.

    Methods:
        name2leaf: Finds a leaf node with the given name.
        find: Alias for name2leaf method.
        name2constraints: Retrieves the constraints of a leaf node with the given name.

    Raises:
        ValueError: If the leaf node with the given name is not found.

    Returns:
        None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make sure all leaf names are unique
        leaves = {}
        for l in self.get_nodes_of_type(Leaf):
            n = l.name
            leaves.setdefault(n, l)
            assert l is leaves[n], f"Duplicate name {n} found in architecture"

    def name2leaf(self, name: str) -> "Leaf":
        """
        Finds a leaf node with the given name.

        Args:
            name (str): The name of the leaf node to find.

        Returns:
            Leaf: The leaf node with the given name.

        Raises:
            ValueError: If the leaf node with the given name is not found.
        """
        if isinstance(self, Leaf) and getattr(self, "name", None) == name:
            return self
        for element in self if isinstance(self, list) else self.values():
            try:
                return element.name2leaf(name)
            except (AttributeError, ValueError):
                pass
        raise ValueError(f"Leaf {name} not found in {self}")

    def find(self, *args, **kwargs) -> "Leaf":
        """
        Alias for name2leaf function.
        """
        return self.name2leaf(*args, **kwargs)

    def name2constraints(self, name: str) -> "constraints.ConstraintGroup":
        """
        Retrieves the constraints of a leaf node with the given name.

        Args:
            name (str): The name of the leaf node.

        Returns:
            constraints.ConstraintGroup: The constraints of the leaf node.
        """
        return self.name2leaf(name).constraints


class ArchNodes(ArchNode, ListNode):
    """
    A collection of architectural nodes.

    This class inherits from `ArchNode` and `ListNode` classes.

    Attributes:
        None

    Methods:
        declare_attrs: Declares attributes for the architectural nodes.
        __init__: Initializes an instance of the `ArchNodes` class.
        combine: Combines two `ArchNodes` instances.
        __repr__: Returns a string representation of the `ArchNodes` instance.
        parse_expressions: Parses expressions in the `ArchNodes` instance.

    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(
            "!Component",
            (Storage, Network, Compute, Nothing, Component),
            callfunc=component_factory,
        )
        super().add_attr("!Container", Container)
        super().add_attr("!Hierarchical", Hierarchical)
        super().add_attr("!Parallel", Parallel)
        super().add_attr("!Pipelined", Pipelined)
        super().add_attr("!Nothing", Nothing)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def combine(self, other: "ArchNodes") -> "ArchNodes":
        """
        Combines two `ArchNodes` instances.

        Args:
            other: Another `ArchNodes` instance to combine with.

        Returns:
            A new `ArchNodes` instance that is the combination of self and other.

        """
        return ArchNodes(self + other)

    def __repr__(self):
        """
        Returns a string representation of the `ArchNodes` instance.

        Returns:
            A string representation of the `ArchNodes` instance.

        """
        return f"{self.__class__.__name__}({super().__repr__()})"

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        """
        Parses expressions in the `ArchNodes` instance.

        Args:
            symbol_table: A dictionary representing the symbol table.
            parsed_ids: A set of parsed IDs.

        Returns:
            The parsed `ArchNodes` instance.

        """
        n_symbol_table = {} if symbol_table is None else symbol_table.copy()
        for l in self.get_nodes_of_type(Leaf):
            n_symbol_table[l.name] = l

        def callfunc(x, sym_table):
            if isinstance(x, Container) and not sym_table.get("_in_parallel", False):
                sym_table.setdefault("_parent_container_attributes", {})
                sym_table.update(x.attributes)
            return x

        return super().parse_expressions(n_symbol_table, parsed_ids, callfunc)


class Branch(ArchNode, DictNode, ABC):
    """
    A branch in the architecture.

    Attributes:
        nodes (ArchNodes): List of child nodes in the branch.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("nodes", ArchNodes, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes: ArchNodes = self["nodes"]

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        n_symbol_table = {} if symbol_table is None else symbol_table.copy()
        n_symbol_table["_in_parallel"] = isinstance(self, Parallel)
        return super().parse_expressions(symbol_table, parsed_ids)


class Hierarchical(Branch):
    """
    A hierarchical branch in the architecture.
    """

    pass


class Parallel(Branch):
    """
    A parallel branch in the architecture.
    """

    pass


class Pipelined(Branch):
    """ "
    A pipelined branch in the architecture.
    """

    pass


class Architecture(Hierarchical):
    """
    An architecture.

    Attributes:
        version (Union[str, Number]): The version of the architecture.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4", callfunc=assert_version)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: Union[str, Number] = self["version"]

    def combine(self, other: "Architecture") -> "Architecture":
        self.logger.warning(
            "Multiple architectures found. Appending the nodes from one arch "
            "to the other. Ignore this warning if this was intended."
        )
        self.nodes += other.nodes
        return self


class Leaf(ArchNode, DictNode, ABC):
    """
    A leaf node in the architecture hierarchy.

    Attributes:
        name (str): The name of the leaf node.
        attributes (Attributes): The attributes associated with the leaf node.
        spatial (Spatial): The spatial attributes of the leaf node.
        constraints (ConstraintGroup): The constraint group associated with the leaf node.
        sparse_optimizations (SparseOptimizationGroup): The sparse optimization group associated with the leaf node.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        # Class named _class to avoid clashing with class keyword
        super().add_attr("attributes", Attributes, {})
        super().add_attr("spatial", Spatial, {})
        super().add_attr("constraints", constraints.ConstraintGroup, {})
        super().add_attr("sparse_optimizations", SparseOptimizationGroup, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.attributes: Attributes = self["attributes"]
        self.spatial: Spatial = self["spatial"]
        self.constraints: constraints.ConstraintGroup = self["constraints"]
        self.sparse_optimizations: SparseOptimizationGroup = self[
            "sparse_optimizations"
        ]

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        """
        Parse the expressions in the leaf node.

        Args:
            symbol_table (Optional[Dict[str, Any]]): The symbol table for parsing expressions.
            parsed_ids (Optional[set]): The set of parsed IDs.

        Returns:
            Attributes: The parsed attributes.
        """
        n_symbol_table = {} if symbol_table is None else symbol_table.copy()

        def callfunc(x, sym_table):
            # Fill the attributes with the parent attributes
            sym_table["attributes"] = {
                **sym_table.get("_parent_container_attributes", {}),
                **sym_table.get("attributes", {}),
            }
            return x

        callfunc(None, n_symbol_table)
        super().parse_expressions(n_symbol_table, parsed_ids)
        return self.attributes

    def get_fanout(self):
        return self.spatial.get_fanout()


class Component(Leaf, ABC):
    """
    A component in the architecture.

    Attributes:
        _class (str): The class of the component.
        subclass (str): The subclass of the component.
        required_actions (List[str]): The list of required actions for the component.
        area_scale (float): How much to scale the area of the component.
        enabled (bool): Indicates whether the component is enabled or not.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("class", str)
        super().add_attr("subclass", str, None)
        super().add_attr("required_actions", list, [])
        super().add_attr("area_scale", Number, None)
        super().add_attr("energy_scale", Number, None)
        super().add_attr("enabled", bool, True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class: str = self["class"]
        self.subclass: str = self["subclass"]
        self.required_actions: List[str] = self["required_actions"]
        self.area_scale: float = self["area_scale"]
        self.energy_scale: float = self["energy_scale"]
        self.enabled: bool = self["enabled"]

    def _check_unrecognized(self, *args, **kwargs):
        return super()._check_unrecognized(*args, **kwargs)


class Container(Leaf, ABC):
    """
    A container in the architecture.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("networks", Networks, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.networks: Networks = self["networks"]


class Networks(ListNode):
    """
    A list of networks in the architecture.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Network)


class Storage(Component):
    """
    A storage component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("attributes", StorageAttributes, {})

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.attributes: StorageAttributes = self["attributes"]


class Compute(Component):
    """
    A compute component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class Network(Component):
    """
    A network component.

    This class inherits from the Component class and provides additional
    functionality specific to networks.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class Spatial(DictNode):
    """
    A spatial configuration in a system architecture.

    Attributes:
        meshX (int): The number of elements in the X dimension of the mesh.
        meshY (int): The number of elements in the Y dimension of the mesh.
        get_fanout (Callable): A function that returns the fanout of the spatial configuration.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("meshX", (int), 1)
        super().add_attr("meshY", (int), 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meshX: int = self["meshX"]
        self.meshY: int = self["meshY"]

    def validate_fanout(self):
        for target in ["meshX", "meshY"]:
            v = self[target]
            assert int(v) == v, f"{target} must be an integer, but is {v}"
            assert v > 0, f"{target} must be positive, but is {v}"

    def get_fanout(self):
        return self.meshX * self.meshY

    def to_fanout_string(self):
        return f"[1..{self.get_fanout()}]"


class Attributes(DictNode):
    """
    A class representing attributes for a node in the architecture.

    Attributes:
        has_power_gating (bool): Indicates whether the node has power gating.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("has_power_gating", (str, bool), False)
        super().add_attr("power_gated_at", str, None)
        super().add_attr("", part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_parse = True


class StorageAttributes(Attributes):
    """Represents the attributes of a storage element.

    This class provides methods to declare and initialize various attributes
    related to a storage element, such as datawidth, technology, number of banks,
    block size, cluster size, etc.

    Attributes:
        datawidth (Union[str, int]): The datawidth of the storage element.
        technology (Union[str, int]): The technology used for the storage element.
        n_banks (Union[str, int]): The number of banks in the storage element.
        block_size (Union[str, Number]): The block size of the storage element.
        cluster_size (Union[str, Number]): The cluster size of the storage element.
        depth (Union[str, Number]): The depth of the storage element.
        entries (Union[str, Number]): The number of entries in the storage element.
        sizeKB (Union[str, Number]): The size of the storage element in kilobytes.
        reduction_supported (Union[str, bool]): Indicates if reduction is supported.
        multiple_buffering (Union[str, Number]): The level of multiple buffering.
        min_utilization (Union[str, Number]): The minimum utilization of the storage element.
        shared_bandwidth (Union[str, Number]): The shared bandwidth of the storage element.
        read_bandwidth (Union[str, Number]): The read bandwidth of the storage element.
        write_bandwidth (Union[str, Number]): The write bandwidth of the storage element.
        network_fill_latency (Union[str, int]): The network fill latency of the storage element.
        network_drain_latency (Union[str, int]): The network drain latency of the storage element.
        allow_overbooking (Union[str, bool]): Indicates if overbooking is allowed.
        metadata_block_size (Union[str, int]): The block size of the metadata storage.
        metadata_datawidth (Union[str, int]): The datawidth of the metadata storage.
        metadata_storage_width (Union[str, int]): The storage width of the metadata storage.
        metadata_storage_depth (Union[str, int]): The storage depth of the metadata storage.
        concordant_compressed_tile_traversal (Union[str, bool]): Indicates if concordant compressed tile traversal is supported.
        tile_partition_supported (Union[str, bool]): Indicates if tile partition is supported.
        decompression_supported (Union[str, bool]): Indicates if decompression is supported.
        compression_supported (Union[str, bool]): Indicates if compression is supported.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        # Attribute declarations...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attribute initialization...


class StorageAttributes(Attributes):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("datawidth", (str, int))
        super().add_attr("technology", (str, int), None)
        super().add_attr("n_banks", (str, int), 2)
        super().add_attr("block_size", (str, int), None)
        super().add_attr("cluster_size", (str, int), 1)
        super().add_attr("width", (str, int), None)
        super().add_attr("depth", (str, int), None)
        super().add_attr("entries", (str, int), None)
        super().add_attr("sizeKB", (str, int), None)
        super().add_attr("reduction_supported", (str, bool), True)
        super().add_attr("multiple_buffering", (str, Number), 1)
        super().add_attr("min_utilization", (str, Number), 0)
        # Bandwidth and latency
        super().add_attr("shared_bandwidth", (str, Number), None)
        super().add_attr("read_bandwidth", (str, Number), None)
        super().add_attr("write_bandwidth", (str, Number), None)
        super().add_attr("network_fill_latency", (str, int), None)
        super().add_attr("network_drain_latency", (str, int), None)
        super().add_attr("per_dataspace_bandwidth_consumption_scale", dict, {})
        # Overbooking
        super().add_attr("allow_overbooking", (str, bool), False)
        # Sparse optimization
        super().add_attr("metadata_block_size", (str, int), None)
        super().add_attr("metadata_datawidth", (str, int), None)
        super().add_attr("metadata_storage_width", (str, int), None)
        super().add_attr("metadata_storage_depth", (str, int), None)
        super().add_attr("concordant_compressed_tile_traversal", (str, bool), None)
        super().add_attr("tile_partition_supported", (str, bool), None)
        super().add_attr("decompression_supported", (str, bool), None)
        super().add_attr("compression_supported", (str, bool), None)

        super().require_one_of("entries", "sizeKB", "depth")
        super().require_one_of("block_size", "cluster_size")
        super().require_all_or_none_of(
            "metadata_datawidth",
            "metadata_storage_width",
            "metadata_storage_depth",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datawidth: Union[str, int] = self["datawidth"]
        self.technology: Union[str, int] = self["technology"]
        self.n_banks: Union[str, int] = self["n_banks"]
        self.block_size: Union[str, Number] = self["block_size"]
        self.cluster_size: Union[str, Number] = self["cluster_size"]
        self.depth: Union[str, Number] = self["depth"]
        self.entries: Union[str, Number] = self["entries"]
        self.sizeKB: Union[str, Number] = self["sizeKB"]
        self.reduction_supported: Union[str, bool] = self["reduction_supported"]
        self.multiple_buffering: Union[str, Number] = self["multiple_buffering"]
        self.min_utilization: Union[str, Number] = self["min_utilization"]
        self.shared_bandwidth: Union[str, Number] = self["shared_bandwidth"]
        self.read_bandwidth: Union[str, Number] = self["read_bandwidth"]
        self.write_bandwidth: Union[str, Number] = self["write_bandwidth"]
        self.network_fill_latency: Union[str, int] = self["network_fill_latency"]
        self.network_drain_latency: Union[str, int] = self["network_drain_latency"]
        self.allow_overbooking: Union[str, bool] = self["allow_overbooking"]
        self.metadata_block_size: Union[str, int] = self["metadata_block_size"]
        self.metadata_datawidth: Union[str, int] = self["metadata_datawidth"]
        self.metadata_storage_width: Union[str, int] = self["metadata_storage_width"]
        self.metadata_storage_depth: Union[str, int] = self["metadata_storage_depth"]
        self.concordant_compressed_tile_traversal: Union[str, bool] = self[
            "concordant_compressed_tile_traversal"
        ]
        self.tile_partition_supported: Union[str, bool] = self[
            "tile_partition_supported"
        ]
        self.decompression_supported: Union[str, bool] = self["decompression_supported"]
        self.compression_supported: Union[str, bool] = self["compression_supported"]


class Nothing(Component):
    """
    A class representing a 'nothing' component.

    This class inherits from the Component class and provides a default implementation
    for the 'nothing' component.

    Attributes:
        name (str): The name of the component.
        class (str): The class of the component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "nothing"
        if "class" not in kwargs:
            kwargs["class"] = "nothing"
        super().__init__(self, *args, **kwargs)


def component_factory(*args, **kwargs) -> "Component":
    """
    Factory function for creating components based on the provided arguments.

    Args:
        *args: Variable length arguments. Either a single dictionary or keyword arguments.
        **kwargs: Keyword arguments. Either a single dictionary or keyword arguments.

    Returns:
        Component: The created component.

    Raises:
        TypeError: If both a dictionary and keyword arguments are provided, or if no dictionary is provided.
        TypeError: If the provided argument is not a dictionary.
        AssertionError: If the 'class' attribute is missing in the provided dictionary.
        AssertionError: If the 'class' attribute is not a string.
        ValueError: If the element class is unknown.

    """
    all_args = list(args) + ([kwargs] if kwargs else [])
    f = "Pass either a dictionary or keyword arguments, but not both."
    if len(all_args) > 1:
        raise TypeError(f"Too many arguments given to component_factory(). {f}")
    if len(all_args) == 0:
        raise TypeError(f"No dictionary given to component_factory(). {f}")
    if not isinstance(all_args[0], dict):
        raise TypeError(f"No dictionary given to component_factory(). {f}")

    kwargs = all_args[0]
    assert "class" in kwargs, f"Component missing 'class' attribute."
    assert isinstance(
        kwargs.get("class", None), str
    ), f'Component "class" attribute must be a string. Got {kwargs["class"]}'
    element_class = kwargs["class"]
    class2class = {
        BUFFER_CLASSES: Storage,
        COMPUTE_CLASSES: Compute,
        NETWORK_CLASSES: Network,
        NOTHING_CLASSES: Nothing,
    }

    for c, target in class2class.items():
        if any([e in element_class for e in c]):
            return target(**kwargs)

    raise ValueError(
        f"Unknown element class {element_class}. " f"Accepted classes: {class2class}"
    )


def dummy_storage(name: str) -> "Storage":
    """
    Create a dummy storage component.

    Args:
        name (str): The name of the storage component.

    Returns:
        Storage: The created dummy storage component.
    """
    attrs = {"width": 1, "depth": 1, "datawidth": 1, "technology": -1}
    args = {"name": name, "class": "dummy_storage", "attributes": attrs}
    return component_factory(**args)


Attributes.declare_attrs()
Spatial.declare_attrs()
Component.declare_attrs()
Storage.declare_attrs()
Compute.declare_attrs()
Network.declare_attrs()
Container.declare_attrs()
ArchNodes.declare_attrs()
Hierarchical.declare_attrs()
Parallel.declare_attrs()
Pipelined.declare_attrs()
Architecture.declare_attrs()
Leaf.declare_attrs()
Networks.declare_attrs()
Branch.declare_attrs()
Nothing.declare_attrs()
StorageAttributes.declare_attrs()
