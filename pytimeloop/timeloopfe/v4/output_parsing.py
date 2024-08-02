import copy
from numbers import Number
import os
from typing import Any, Dict, Tuple, List, Union
import yaml


def parse_stats_file(path: str) -> Tuple[int, int, float, dict]:
    """
    Parse a stats file from Timeloop.
    Args:
        path (str): The path to the stats file.

    Returns:
        Tuple[int, int, float, dict]: The cycles, computes, percent utilization, and energy.

    """
    lines = open(path, "r").readlines()
    cycles, computes, util, energy = None, None, None, {}
    for i, l in enumerate(lines):
        if "Computes =" in l:
            computes = int(l.split()[-1])
            break
        if "Cycles: " in l:
            cycles = int(l.split()[-1])
        if "Utilization" in l:
            util = float(l.split()[-1][:-1]) / 100

    assert cycles is not None, f"Could not find cycles in stats at {path}."
    assert computes is not None, f"Could not find computes in stats at {path}."
    assert util is not None, f"Could not find percent_utilization in stats at {path}."

    for l in lines[i + 1 :]:
        if "=" in l:
            e, v = l.rsplit("=", 1)
            if e.strip() == "Total":
                continue
            energy[e.strip()] = float(v.strip()) * computes / 1e15

    return cycles, computes, util, energy


def get_area_from_art(path: str) -> dict:
    """
    Get the area of each component from an ART file.

    Args:
        path (str): The path to the ART file.

    Returns:
        dict: The area of each component.

    """
    d = yaml.load(open(path, "r").read(), Loader=yaml.SafeLoader)
    name2area = {}
    for x in d["ART"]["tables"]:
        namecount = x["name"].split(".", 1)[1]
        name = namecount.split("[", 1)[0]
        count = (
            int(namecount.split("[", 1)[1].split(".")[-1][:-1])
            if "[" in namecount
            else 1
        )
        name2area[name] = count * x["area"] / 1e12
    return name2area


class MultipliableDict(dict):
    """
    A dictionary that can be multiplied or divided by a scalar.
    """

    def __truediv__(self, other):
        return MultipliableDict({k: v / other for k, v in self.items()})

    def __mul__(self, other):
        return MultipliableDict({k: v * other for k, v in self.items()})

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)


class OutputStats:
    """
    A class to store the output statistics from Timeloop.

    Parameters:
        percent_utilization (float): The utilization percentage.
        computes (int): The number of computes.
        cycles (int): The number of cycles.
        cycle_seconds (float): The duration of a cycle in seconds.
        per_component_energy (Dict[str, float]): The energy consumed by each component in Joules.
        per_component_area (Dict[str, float]): The area of each component in square meters.
        variables (dict): The variables used in the specification.
        mapping (str): The mapping result.
    """

    def __init__(
        self,
        percent_utilization: float,
        computes: int,
        cycles: int,
        cycle_seconds: float,
        per_component_energy: Dict[str, float],
        per_component_area: Dict[str, float],
        variables: dict,
        mapping: str = "",
    ):
        self.percent_utilization: float = percent_utilization
        self.computes: int = computes
        self.cycles: int = cycles
        self.cycle_seconds: float = cycle_seconds
        self.latency: float = cycles * cycle_seconds
        self.per_component_energy: Dict[str, float] = MultipliableDict(
            **per_component_energy
        )
        self.per_component_area: Dict[str, float] = MultipliableDict(
            **per_component_area
        )
        self.variables: dict = copy.deepcopy(variables)
        for k, v in self.variables.items():
            # These can't pickle, so we'll just store the name to allow OutputStats to
            # be pickled and returned from subprocesses.
            if callable(v):
                self.variables[k] = f"function {v.__name__}"

        self.area: float = sum(per_component_area.values())
        self.energy: float = sum(per_component_energy.values())
        self.computes_per_second: float = computes / cycle_seconds / cycles
        self.computes_per_second_per_square_meter: float = (
            self.computes_per_second / self.area
        )
        self.computes_per_joule: float = self.computes / self.energy
        self.mapping: str = mapping

    def scale_computes_by(self, factor: float):
        self.computes *= factor
        self.computes_per_second *= factor
        self.computes_per_joule *= factor

    @staticmethod
    def aggregate(tests: List["OutputStats"]) -> "OutputStats":
        """
        Aggregate a list of OutputStats into a single OutputStats object.

        Args:
            tests (List[OutputStats]): A list of OutputStats objects to aggregate.
        """
        results = {}

        for grab_last in ["cycle_seconds", "per_component_area", "variables"]:
            results[grab_last] = getattr(tests[-1], grab_last)

        for to_sum in ["computes", "cycles"]:
            results[to_sum] = sum(getattr(t, to_sum) for t in tests)

        # Sum
        results["per_component_energy"] = MultipliableDict({})
        for t in tests:
            for k, v in t.per_component_energy.items():
                results["per_component_energy"][k] = (
                    results["per_component_energy"].get(k, 0) + v
                )

        # Weighted average
        results["percent_utilization"] = (
            sum(t.percent_utilization * t.computes for t in tests) / results["computes"]
        )

        results["mapping"] = None

        return OutputStats(**results)

    def access(self, key: str) -> Any:
        """
        Access a key in the OutputStats object. If the key is not found, check
        the variables.

        Args:
            key (str): The key to access.

        Returns:
            Any: The value of the key.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            if key in self.variables:
                return self.variables[key]
            raise AttributeError(
                f"Could not find key {key}. Available keys: "
                f"{self.__annotations__} and {list(self.variables.keys())}"
            )

    @staticmethod
    def aggregate_by(
        tests: List["OutputStats"], *keys: Union[List[str], str]
    ) -> "OutputStatsList":
        """
        Aggregate a list of OutputStats objects by a set of keys. OutputStats with
        equal values for the keys will be aggregated together. OutputStats with
        different values for the keys will be aggregated separately and returned
        as a list.

        Args:
            tests (List[OutputStats]): A list of OutputStats objects to aggregate.
            keys (List[str]): The keys to aggregate by.

        Returns:
            OutputStatsList: A list of aggregated OutputStats objects.

        """
        to_agg = {}
        for t in tests:
            key = tuple(t.access(k) for k in keys)
            to_agg[key] = to_agg.get(key, []) + [t]

        return OutputStatsList(OutputStats.aggregate(v) for v in to_agg.values())

    def combine_per_component_area(self, from_keys: List[str], to: str):
        """
        Combine the area of multiple components into a single component.

        Args:
            from_keys (List[str]): The keys of the components to combine.
            to (str): The key to combine the components into.
        """
        if not all(k in self.per_component_area for k in from_keys):
            raise KeyError(
                f"Could not find all keys {from_keys} in per_component_area. "
                f"Keys: {self.per_component_area.keys()}"
            )
        assert len(set(from_keys)) == len(from_keys), (
            f"Duplicate keys found in {from_keys}. "
            f"Keys: {self.per_component_area.keys()}"
        )
        self.per_component_area[to] = sum(
            self.per_component_area.pop(k) for k in from_keys
        )

    def combine_per_component_energy(self, from_keys: List[str], to: str):
        if not all(k in self.per_component_energy for k in from_keys):
            raise KeyError(
                f"Could not find all keys {from_keys} in per_component_energy. "
                f"Keys: {self.per_component_energy.keys()}"
            )
        assert len(set(from_keys)) == len(from_keys), (
            f"Duplicate keys found in {from_keys}. "
            f"Keys: {self.per_component_energy.keys()}"
        )
        self.per_component_energy[to] = sum(
            self.per_component_energy.pop(k) for k in from_keys
        )

    def combine_per_component_area_energy(self, from_keys: List[str], to: str):
        """
        Combine the area and energy of multiple components into a single component.

        Args:
            from_keys (List[str]): The keys of the components to combine.
            to (str): The key to combine the components into.
        """
        self.combine_per_component_area(from_keys, to)
        self.combine_per_component_energy(from_keys, to)

    def clear_zero_energies(self):
        """
        Remove components with zero energy.
        """
        for k in list(self.per_component_energy.keys()):
            if self.per_component_energy[k] == 0:
                del self.per_component_energy[k]

    def clear_zero_areas(self):
        """
        Remove components with zero area.
        """
        for k in list(self.per_component_area.keys()):
            if self.per_component_area[k] == 0:
                del self.per_component_area[k]

    def per_compute(self, key: str) -> Union[float, MultipliableDict]:
        """
        Returns a value scaled by the number of computes.

        Args:
            key (str): The key to access.

        Returns:
            Union[float, MultipliableDict]: The scaled value.
        """
        if key == "per_component_energy" or key == "per_component_area":
            d = getattr(self, key)
            return MultipliableDict(**{k: v / self.computes for k, v in d.items()})
        return getattr(self, key) / self.computes


class OutputStatsList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def combine_per_component_area_energy(self, from_keys: List[str], to: str):
        """
        Combine the area and energy of multiple components into a single component.

        Args:
            from_keys (List[str]): The keys of the components to combine.
            to (str): The key to combine the components into.
        """
        for t in self:
            t.combine_per_component_area_energy(from_keys, to)

    def combine_per_component_area(self, from_keys: List[str], to: str):
        """
        Combine the area of multiple components into a single component.

        Args:
            from_keys (List[str]): The keys of the components to combine.
            to (str): The key to combine the components into.
        """
        for t in self:
            t.combine_per_component_area(from_keys, to)

    def combine_per_component_energy(self, from_keys: List[str], to: str):
        """
        Combine the energy of multiple components into a single component.

        Args:
            from_keys (List[str]): The keys of the components to combine.
            to (str): The key to combine the components into.
        """
        for t in self:
            t.combine_per_component_energy(from_keys, to)

    def aggregate(self) -> OutputStats:
        """
        Aggregate the OutputStats objects in the list.

        Returns:
            OutputStats: The aggregated OutputStats object.
        """
        return OutputStats.aggregate(self)

    def aggregate_by(self, *keys: str) -> "OutputStatsList":
        """
        Aggregate the OutputStats objects in the list by a set of keys. OutputStats with
        equal values for the keys will be aggregated together. OutputStats with
        different values for the keys will be aggregated separately and returned
        as a list.

        Args:
            keys (List[str]): The keys to aggregate by.

        Returns:
            OutputStatsList: A list of aggregated OutputStats objects.
        """
        return OutputStatsList(OutputStats.aggregate_by(self, *keys))

    def split_by(self, *keys: str) -> List["OutputStatsList"]:
        """
        Split the OutputStats objects in the list by a set of keys. Returns a
        list of OutputStatsList objects, where each OutputStatsList contains the
        OutputStats objects with the same values for the keys.

        Args:
            keys (List[str]): The keys to split by.

        Returns:
            List[OutputStatsList]: A list of OutputStatsList objects.
        """
        to_agg = {}
        for t in self:
            key = tuple(t.access(k) for k in keys)
            to_agg[key] = to_agg.get(key, []) + [t]

        return [OutputStatsList(v) for v in to_agg.values()]

    def clear_zero_energies(self):
        """
        Remove components with zero energy.
        """
        for t in self:
            t.clear_zero_energies()

    def clear_zero_areas(self):
        """
        Remove components with zero area.
        """
        for t in self:
            t.clear_zero_areas()


def parse_timeloop_output(
    spec: "Specification",
    output_dir: str,
    prefix: str,
) -> OutputStats:
    """
    Parse the output of Timeloop.

    Args:
        spec (Specification): The Timeloop specification.
        output_dir (str): The output directory.
        prefix (str): The prefix of the output files.

    Returns:
        OutputStats: The parsed output statistics.

    """
    stats_path = os.path.join(output_dir, f"{prefix}.stats.txt")
    art_path = os.path.join(output_dir, f"{prefix}.ART.yaml")

    cycles, computes, percent_utilization, energy = parse_stats_file(stats_path)
    area = get_area_from_art(art_path)

    for k in list(area.keys()) + list(energy.keys()):
        area.setdefault(k, 0)
        energy.setdefault(k, 0)

    spec.parse_expressions()
    mapping = None
    if os.path.exists(stats_path.replace(".stats.txt", ".map.txt")):
        mapping = open(stats_path.replace(".stats.txt", ".map.txt")).read()

    try:
        cycle_seconds = spec.variables["GLOBAL_CYCLE_SECONDS"]
    except:
        cycle_seconds = 1e-9
        from .arch import Leaf

        for s in spec.architecture.get_nodes_of_type(Leaf):
            c = s.attributes.get("GLOBAL_CYCLE_SECONDS", None)
            if isinstance(c, Number):
                cycle_seconds = c
                break

    return OutputStats(
        percent_utilization=percent_utilization,
        computes=computes,
        cycles=cycles,
        cycle_seconds=cycle_seconds,
        per_component_energy=energy,
        per_component_area=area,
        variables=spec.variables,
        mapping=mapping,
    )
