import logging
from typing import Iterable, List, Optional, Tuple, Type, Union
from ..common.nodes import DictNode, ListNode, isempty, CombinableListNode
import pytimeloop.timeloopfe.v4.problem as problem
import copy
from .version import assert_version


def dummy_constraints(
    prob: problem.Problem, create_spatial_constraint: bool = False
) -> "ConstraintGroup":
    """
    Creates a dummy constraint group for the given problem.

    Args:
        prob (problem.Problem): The problem for which the constraints are created.
        create_spatial_constraint (bool, optional): Whether to create a spatial constraint. Defaults to False.

    Returns:
        ConstraintGroup: The created constraint group.
    """
    c = ConstraintGroup()
    c.temporal = Temporal(
        factors=list(f"{x}=1" for x in prob.shape.dimensions),
        permutation=copy.deepcopy(prob.shape.dimensions),
    )
    c.dataspace = Dataspace(bypass=[x.name for x in prob.shape.data_spaces])
    if create_spatial_constraint:
        c.spatial = Spatial(
            factors=c.temporal.factors,
            permutation=c.temporal.permutation,
        )
    return c


def constraint_factory(constraint: dict):
    """
    Factory function to create constraint objects based on the provided dictionary.

    Args:
        constraint (dict): A dictionary containing the constraint information.

    Returns:
        object: An instance of the appropriate constraint class based on the 'type' field in the dictionary.

    Raises:
        ValueError: If the 'type' field is missing or not recognized.
    """
    if "type" not in constraint:
        raise ValueError("Constraint must have a type")
    ctype = constraint["type"]
    type2class = {
        "spatial": Spatial,
        "temporal": Temporal,
        "dataspace": Dataspace,
        "max_overbooked_proportion": MaxOverbookedProportion,
        "utilization": Utilization,
    }
    if ctype not in type2class:
        raise ValueError(
            f"Constraint type '{ctype}' not recognized."
            f"Must be one of {list(type2class.keys())}"
        )
    return type2class[ctype](**constraint)


class Constraints(DictNode):
    """
    Class representing constraints.

    Attributes:
        version (str): The version of the constraints.
        targets (ConstraintsList): The list of targets for the constraints.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.4", callfunc=assert_version)
        super().add_attr("targets", ConstraintsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.targets: ConstraintsList = self["targets"]


class ConstraintsList(CombinableListNode):
    """
    A class representing a list of constraints.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(
            "",
            (Spatial, Temporal, Dataspace, MaxOverbookedProportion),
            callfunc=constraint_factory,
        )


class Constraint(DictNode):
    """
    A constraint in the system.

    Args:
        type (str): The type of the constraint.
        target (str): The target of the constraint.

    Attributes:
        type (str): The type of the constraint.
        target (str): The target of the constraint.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("type", str, "")
        super().add_attr("target", str, "")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = self["type"]
        self.target: str = self["target"]
        self._disjoint_dataspaces_lists = []

    def isempty(self):
        return all(isempty(v) for k, v in self.items() if k not in ["type", "target"])

    def clear_respecification(self, other: "Constraint", mine: dict, others: dict):
        overlapping = set(mine.keys()) & set(others.keys())
        overlapping_not_equal = {k: mine[k] != others[k] for k in overlapping}
        problems = [
            f"{k}={mine[k]} AND {k}={others[k]}"
            for k, v in overlapping_not_equal.items()
            if v
        ]

        if problems:
            raise ValueError(
                f"Re-specification of {problems} in two "
                f"{self.__class__.__name__} constraints:\n{self}\nand {other}."
            )
        return mine, others

    def list_attrs_to_dict(self, attrs: List[str]) -> dict:
        flattened = {}
        for a in attrs:
            for k in self.get(a, []):
                if k in flattened and flattened[k] != a:
                    raise ValueError(
                        f"Re-specification of {k} found in {attrs} for "
                        f"constraint {self.__class__.__name__}."
                    )
                flattened[k] = a
        return flattened

    def set_list_attrs_from_dict(
        self, d: dict, attrs: Iterable[str] = (), cast_to_type: Type = list
    ) -> dict:
        lists = {k: cast_to_type() for k in attrs}
        for k, v in d.items():
            lists.setdefault(v, []).append(k)
        self.update(lists)

    def combine_list_attrs(self, other: "Constraint", attrs: List[str]) -> "Constraint":
        if attrs[0] in self:
            mytype = type(self[attrs[0]])
        elif attrs[0] in other:
            mytype = type(other[attrs[0]])
        else:
            return self
        mine = self.list_attrs_to_dict(attrs)
        others = other.list_attrs_to_dict(attrs)
        a, b = self.clear_respecification(other, mine, others)
        self.set_list_attrs_from_dict({**a, **b}, attrs, mytype)
        other.set_list_attrs_from_dict({}, attrs, mytype)
        return self

    def combine(self, other: "Constraint") -> "Constraint":  # Override
        if self.type != other.type:
            raise ValueError("Cannot combine constraints of different types.")
        for ds in self._disjoint_dataspaces_lists:
            self.combine_list_attrs(other, ds)
        return super().combine(other)

    def __str__(self):
        return f"{self.type} constraint(target={self.target}) {super().__str__()}"


class ConstraintGroup(DictNode):
    """
    A group of constraints.

    Attributes:
        spatial (Spatial): The spatial constraint.
        temporal (Temporal): The temporal constraint.
        dataspace (Dataspace): The dataspace constraint.
        max_overbooked_proportion (MaxOverbookedProportion): The maximum overbooked proportion constraint.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("spatial", Spatial, {})
        super().add_attr("temporal", Temporal, {})
        super().add_attr("dataspace", Dataspace, {})
        super().add_attr("max_overbooked_proportion", MaxOverbookedProportion, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial: Spatial = self["spatial"]
        self.temporal: Temporal = self["temporal"]
        self.dataspace: Dataspace = self["dataspace"]
        self.max_overbooked_proportion: MaxOverbookedProportion = self[
            "max_overbooked_proportion"
        ]


class Iteration(Constraint):
    """
    An iteration (spatial or temporal) constraint.

    Attributes:
        factors (Factors): The factors associated with the iteration.
        permutation (Permutation): The permutation associated with the iteration.
        default_max_factor (int): The default maximum factor value.
        default_min_factor (int): The default minimum factor value.
        remainders (int): The remainders associated with the iteration.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("factors", Factors, [], Factors.factory)
        super().add_attr("permutation", Permutation, [], Permutation.factory)
        super().add_attr("default_max_factor", int, None)
        super().add_attr("default_min_factor", int, None)
        super().add_attr("remainders", int, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factors: Factors = self["factors"]
        self.permutation: Permutation = self["permutation"]
        self.default_max_factor: Optional[int] = self["default_max_factor"]
        self.default_min_factor: Optional[int] = self["default_min_factor"]
        self.remainders: int = self["remainders"]


class Spatial(Iteration):
    """
    A spatial iteration constraint.

    Attributes:
        no_reuse (List[str]): A list of problem dataspaces that should not be reused.
        no_link_transfer (List[str]): A list of problem dataspaces that should not have link transfers.
        split (int): The number of splits for the spatial iteration.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("no_reuse", ProblemDataspaceList, [])
        super().add_attr("no_link_transfer", ProblemDataspaceList, [])
        super().add_attr("split", int, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = "spatial"
        self.no_reuse: List[str] = self["no_reuse"]
        self.no_link_transfer: List[str] = self["no_link_transfer"]
        self.split: int = self["split"]
        self._disjoint_dataspaces_lists.append(("no_reuse",))
        self._disjoint_dataspaces_lists.append(("no_link_transfer",))


class Temporal(Iteration):
    """
    A temporal iteration constraint.

    Attributes:
        no_reuse (List[str]): A list of problem dataspaces that should not be reused.
        rmw_first_update (List[str]): A list of problem dataspaces that should have RMW for the first update (rather than a write only).
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("no_reuse", ProblemDataspaceList, [])
        super().add_attr("rmw_first_update", ProblemDataspaceList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = "temporal"
        self.no_reuse: List[str] = self["no_reuse"]
        self.rmw_first_update: List[str] = self["rmw_first_update"]
        self._disjoint_dataspaces_lists.append(("no_reuse",))


class Dataspace(Constraint):
    """
    A constraint class for specifying dataspace properties.

    Attributes:
        bypass (List[str]): List of bypass dataspace names.
        keep (List[str]): List of keep dataspace names.
        no_coalesce (List[str]): List of no_coalesce dataspace names.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("bypass", ProblemDataspaceList, [])
        super().add_attr("keep", ProblemDataspaceList, [])
        super().add_attr("no_coalesce", ProblemDataspaceList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = "dataspace"
        self.bypass: List[str] = self["bypass"]
        self.keep: List[str] = self["keep"]
        self.no_coalesce: List[str] = self["no_coalesce"]
        self._disjoint_dataspaces_lists.append(("bypass", "keep"))


class MaxOverbookedProportion(Constraint):
    """
    A constraint that defines the maximum overbooked proportion.

    Attributes:
        proportion (float): The maximum overbooked proportion.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("proportion", float, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = "max_overbooked_proportion"
        self.proportion: Optional[float] = self["proportion"]


class Utilization(Constraint):
    """
    A constraint that defines the utilization of a component.

    Attributes:
        min (float or str): The minimum utilization value.
        type (str): The type of the constraint, which is "utilization".
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("min", (float, str), None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = "utilization"


class Permutation(ListNode):
    """
    A permutation of ranks.
    """

    @staticmethod
    def factory(x: Union[str, list]) -> "Permutation":
        if isinstance(x, str):
            logging.warning(
                'Permutation given as string "%s". Trying to turn into a ' "list.",
                str(x),
            )
            if "," in x:
                x = x.split(",")
            else:
                x = [y for y in x]
            x = [y.strip() for y in x if y]
        return Permutation(x)

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)


class Factor(str):
    """A loop factor (e.g., P=1)"""

    pass


class Factors(CombinableListNode):
    """
    A list of factors used to describe loop bounds
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Factor, callfunc=Factors.check_valid_factor)

    @staticmethod
    def check_valid_factor(f):
        """
        Check if a factor is valid.

        Parameters:
        - f: The factor to be checked.

        Returns:
        - Factor: The factor if it is valid.

        Raises:
        - ValueError: If the factor cannot be split.

        """
        if not isinstance(f, str):
            assert isinstance(f, Iterable), f"Expected string or iterable, got {f}."
            f = "".join(str(s) for s in f)
        try:
            Factors.splitfactor(f)
            return Factor(f)
        except ValueError:
            return f

    @staticmethod
    def factory(x: Union[str, list]) -> "Factors":
        """
        Create a Factors object from a string or a list.

        Args:
            x (Union[str, list]): The input string or list.

        Returns:
            Factors: The Factors object created from the input.

        Raises:
            None

        """
        if isinstance(x, str):
            logging.warning(
                'Factors given as string "%s". Trying to turn into a list.',
                str(x),
            )
            if "," in x:
                x = x.split(",")
            elif " " in x:
                x = x.split(" ")
            else:
                x = [x]

        return Factors([y for y in x if y])

    @staticmethod
    def splitfactor(x: str) -> Tuple[str, str, int]:
        """
        Split a factor string into its components.

        Args:
            x (str): The factor string to be split.

        Returns:
            Tuple[str, str, int]: A tuple containing the factor name, the comparison operator, and the factor value.

        Raises:
            ValueError: If none of the valid comparison operators are found in the factor string.

        Example:
            splitfactor("X=123") returns ("X", "=", 123)
        """
        checks = ["<=", ">=", "="]
        for to_check in checks:
            if to_check not in x:
                continue
            a, c = x.split(to_check)
            return a, to_check, int(c)
        raise ValueError(
            f'Did not find any of {checks} in factor "{x}".'
            f'Format each factor as "X=123", "X<=123", or "X>=123". '
            f"Multiple factors may be given as a comma-separated string, "
            f"a space-separated string, or a list of strings."
        )

    def get_split_factors(self) -> List[Tuple[str, str, str]]:
        """
        Get a list of split factors.

        Example:
            get_split_factors() returns [("X", "=", "123"), ("Y", "=", "456"), ("Z", "=", "789")] if the factors are "X=123", "Y=456", and "Z=789".

        """
        return [self.splitfactor(x) for x in self]

    def get_factor_names(self) -> List[str]:
        """
        Get a list of factor names.

        Example:
            get_factor_names() returns ["X", "Y", "Z"] if the factors are "X=123", "Y=456", and "Z=789".

        """
        return [self.splitfactor(x)[0] for x in self]

    def remove_factor(self, name: str):
        """
        Remove a factor from the list of factors.

        Args:
            name (str): The name of the factor to be removed.

        Raises:
            None

        Example:
            remove_factor("X") removes the factor "X=123" from the list of factors.

        """
        for i in range(len(self) - 1, -1, -1):
            if name == self.splitfactor(self[i])[0]:
                self.pop(i)

    def add_eq_factor(self, name: str, value: int, overwrite: bool = False):
        """
        Adds an equality factor to the constraint.

        Args:
            name (str): The name of the factor.
            value (int): The value of the factor.
            overwrite (bool, optional): If True, removes any existing factor with the same name before adding the new one. Defaults to False.
        """
        if overwrite:
            self.remove_factor(name)
        self.append(f"{name}={value}")
        self.check_unique_remove_repeat()

    def add_leq_factor(self, name: str, value: int, overwrite: bool = False):
        """
        Adds a less than or equal to (<=) factor constraint to the list of constraints.

        Args:
            name (str): The name of the factor.
            value (int): The value of the factor.
            overwrite (bool, optional): If True, removes any existing factor with the same name before adding the new one. Defaults to False.
        """
        if overwrite:
            self.remove_factor(name)
        self.append(f"{name}<={value}")
        self.check_unique_remove_repeat()

    def add_geq_factor(self, name: str, value: int, overwrite: bool = False):
        """
        Adds a greater-than-or-equal factor constraint to the constraint set.

        Args:
            name (str): The name of the factor.
            value (int): The value of the factor.
            overwrite (bool, optional): If True, removes any existing factor with the same name before adding the new one. Defaults to False.
        """
        if overwrite:
            self.remove_factor(name)
        self.append(f"{name}>={value}")
        self.check_unique_remove_repeat()

    def _check_factors_compatible(self, a, b):
        a_n, a_eq, a_v = self.splitfactor(a)
        b_n, b_eq, b_v = self.splitfactor(b)
        if a_n != b_n:
            return True
        if a_v == b_v:
            return True
        if a_eq == ">=" and b_eq == "<=":
            return True
        if a_eq == "<=" and b_eq == ">=":
            return True
        return False

    def check_unique_remove_repeat(self):
        unique = set()
        # Identical factors OK
        for i in range(len(self) - 1, -1, -1):
            if self[i] in unique:
                self.pop(i)
            unique.add(self[i])

        # Non-identical, but same name, not OK
        checked = {}
        for i, f in enumerate(self):
            self[i] = self.check_valid_factor(f)
            try:
                name, eq, v = self.splitfactor(f)
            except ValueError:
                continue
            if name not in checked:
                checked[name] = f
                continue
            if self._check_factors_compatible(checked[name], f):
                continue
            raise ValueError(
                f"Found conflicting constraints {f} and {checked[name]} "
                f"for the same variable {name} in {self}."
            )

    def combine(self, other: "Factors") -> "Factors":
        super().combine(other)
        self.check_unique_remove_repeat()
        return self

    def get_minimum_product(self, problem_instance: problem.Instance):
        """
        Calculates the minimum product of all factors in this list.

        Args:
            problem_instance (problem.Instance): The problem instance.

        Returns:
            int: The calculated minimum product.
        """
        allocated = 1
        for f in self:
            dim, comparator, value = self.splitfactor(f)
            value = int(value)
            if value == 0:
                value = problem_instance[dim]
            if comparator == "=" and int(value):
                allocated *= int(value)
        return allocated

    def add_eq_factor_iff_not_exists(self, name: str, value: int) -> bool:
        """
        Add an "name=value" factor iff "name" is not already in the factor list. Return True if the factor was added.
        """
        if name not in self.get_factor_names():
            self.add_eq_factor(name, value)
            return True
        return False

    def add_leq_factor_iff_not_exists(self, name: str, value: int) -> bool:
        """
        Add an "name<=value" factor iff "name" is not already in the factor list. Return True if the factor was added.
        """
        if name not in self.get_factor_names():
            self.add_leq_factor(name, value)
            return True
        return False

    def name2factor(self, name: str) -> Optional[str]:
        """
        Return the factor with the given name, or None if not found.
        """
        for f in self:
            if name == self.splitfactor(f)[0]:
                return f
        raise ValueError(f"Factor {name} not found in {self}.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_unique_remove_repeat()


class ProblemDataspaceList(ListNode):
    """
    A list of problem dataspaces.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Override the in operator
    def __contains__(self, item):
        return super().__contains__(item) or super().__contains__("*")


Constraints.declare_attrs()
Constraint.declare_attrs()
ConstraintGroup.declare_attrs()
Iteration.declare_attrs()
Spatial.declare_attrs()
Temporal.declare_attrs()
Dataspace.declare_attrs()
MaxOverbookedProportion.declare_attrs()
ConstraintsList.declare_attrs()
Factors.declare_attrs()
Utilization.declare_attrs()
Permutation.declare_attrs()
ProblemDataspaceList.declare_attrs()
