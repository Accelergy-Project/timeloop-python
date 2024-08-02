from numbers import Number
from ..common.nodes import ListNode, DictNode
from typing import List, Set, Union
from .version import assert_version


class Problem(DictNode):
    """
    The top-level problem object in Timeloop.

    Attributes:
        version (str): The version of the problem.
        instance (Instance): The instance object for the problem.
        shape (Shape): The shape object for the problem.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4", callfunc=assert_version)
        super().add_attr("instance", Instance)
        super().add_attr("shape", Shape)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.instance: Instance = self["instance"]
        self.shape: Shape = self["shape"]


class Shape(DictNode):
    """
    Problem shape object.

    Attributes:
        name (str): The name of the shape.
        dimensions (ListNode): The dimensions of the shape.
        data_spaces (ProblemDataspaceList): The data spaces of the shape.
        coefficients (ListNode): The coefficients of the shape.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str, "")
        super().add_attr("dimensions", ListNode)
        super().add_attr("data_spaces", ProblemDataspaceList)
        super().add_attr("coefficients", ListNode, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.dimensions: ListNode = self["dimensions"]
        self.data_spaces: DataSpace = self["data_spaces"]
        self.coefficients: ListNode = self["coefficients"]

    def name2dataspace(
        self, name: Union[str, List[str]]
    ) -> Union["DataSpace", List["DataSpace"]]:
        """
        Convert the name of a data space to the corresponding DataSpace object(s).

        Args:
            name (Union[str, List[str]]): The name(s) of the data space(s) to convert.

        Returns:
            Union[DataSpace, List[DataSpace]]: The corresponding DataSpace object(s).

        Raises:
            ValueError: If the data space with the given name is not found.
        """
        if isinstance(name, List):
            return [self.name2dataspace(n) for n in name]
        if isinstance(name, DataSpace):
            return name
        for ds in self.data_spaces:
            if ds.name == name:
                return ds
        raise ValueError(f"Data space {name} not found")

    def dataspace2dims(self, name: Union[str, List[str]]) -> List[str]:
        """
        Convert the name(s) of data space(s) to the corresponding dimensions.

        Args:
            name (Union[str, List[str]]): The name(s) of the data space(s) to convert.

        Returns:
            List[str]: The corresponding dimensions.

        Raises:
            ValueError: If the data space with the given name is not found.
        """
        ds = self.name2dataspace(name)
        factors = set()
        dimensions = set(self.dimensions)
        for d in ds if isinstance(ds, list) else [ds]:
            factors.update(set(d.factors) & dimensions)
        return list(factors)

    def dataspace2unique_dims(self, name: str) -> List[str]:
        """
        Get the unique dimensions associated with a specific data space.

        Args:
            name (str): The name of the data space.

        Returns:
            List[str]: The unique dimensions associated with the data space.
        """
        ds = self.name2dataspace(name)
        ds = ds if isinstance(ds, list) else [ds]
        other_ds = [d for d in self.data_spaces if d not in ds]
        return list(set(self.dataspace2dims(name)) - set(self.dataspace2dims(other_ds)))


class ProblemDataspaceList(ListNode):
    """
    A list of data spaces in the problem.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", DataSpace)


class Instance(DictNode):
    """
    An problem instance object.

    Attributes:
        name (str): The name of the instance.
        data_spaces (DataSpaceList): The data spaces of the instance.
        densities (DensityList): The densities of each data space.

    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("densities", DensityList, {})
        super().add_attr("", int, part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DataSpace(DictNode):
    """
    A data space object.

    Attributes:
        name (str): The name of the data space.
        projection (list): The projection of the data space.
        read_write (str, bool, int): The read-write attribute of the data space.
        factors (list): The factors derived from the projection.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("projection", list)
        super().add_attr("read_write", (str, bool, int), False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name: str = self["name"]
        self.projection: list = self["projection"]
        self.factors: list = []

        projection = [x for x in self.projection]
        while projection:
            factor = projection.pop(0)
            if isinstance(factor, list):
                projection += factor
            else:
                self.factors.append(factor)


class DensityList(DictNode):
    """
    A list of densities for each data space.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Density, part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Density(DictNode):
    """
    A Density object represents the density and distribution of a workload tensor.

    Attributes:
        density (Number or str): The density of the workload tensor.
        distribution (str): The distribution type of the workload tensor.
        band_width (int): The band width of the workload tensor (for banded distribution).
        workload_tensor_size (int): The size of the workload tensor (for hypergeometric distribution).
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("density", (Number, str))
        super().add_attr(
            "distribution",
            ("fixed_structured", "hypergeometric", "banded"),
        )
        super().add_attr("band_width", int, 0)  # Banded
        super().add_attr("workload_tensor_size", int, 0)  # Hypergeometric

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density: Number = self["density"]
        self.distribution: str = self["distribution"]


Problem.declare_attrs()
Shape.declare_attrs()
Instance.declare_attrs()
DataSpace.declare_attrs()
ProblemDataspaceList.declare_attrs()
DensityList.declare_attrs()
Density.declare_attrs()
