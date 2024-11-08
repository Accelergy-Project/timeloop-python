from dataclasses import dataclass


@dataclass
class MacArrayConstraint:
    array_shape_in_parallel_dimension: str
    array_shape_in_reduced_dimension: str

    weight_tensor: dict[str, str]
    parallel_rank: dict[str, str]
    reduced_rank: dict[str, str]


@dataclass
class PeArrayConstraint:
    array_shape: int

