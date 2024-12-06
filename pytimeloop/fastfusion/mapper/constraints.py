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


WILDCARD = '*'
SEPARATOR = '/'

@dataclass
class DataflowConstraint:
    einsum_to_constraint: dict[int, list]

    @staticmethod
    def parse(pattern: dict[str, list[str]], workload):
        einsum_name_to_id = workload.einsum_name_to_id()
        rank_name_to_id = workload.dimension_name_to_id()

        def str_to_rank_or_wildcard(string: str):
            if string == WILDCARD:
                return WILDCARD
            elif string == '/':
                return SEPARATOR
            else:
                return rank_name_to_id[string]

        constraint = {
            einsum_name_to_id[einsum_name]: [
                str_to_rank_or_wildcard(rank_name)
                for rank_name in constraint
            ]
            for einsum_name, constraint in pattern.items()
        }
        for einsum_id in workload.einsum_id_to_name():
            if einsum_id not in constraint:
                constraint[einsum_id] = [WILDCARD]
        return constraint

    @staticmethod
    def default(workload):
        constraint = {}
        for einsum_id in workload.einsum_id_to_name():
            constraint[einsum_id] = [WILDCARD]
        return constraint