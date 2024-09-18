from dataclasses import dataclass


@dataclass
class CompatibilityInfo:
    fused_loops: list[tuple[str, int]]
    fused_tensor: set[str]
    is_pipelined: bool

    def matches(self, other: 'CompatibilityInfo'):
        raise NotImplementedError()

    def combine(self, other: 'CompatibilityInfo') -> 'CompatibilityInfo':
        raise NotImplementedError()


class Payload:
    def __init__(self):
        pass

    def combine(self, other: 'Payload'):
        raise NotImplementedError()
