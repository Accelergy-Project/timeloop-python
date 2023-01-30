import bindings
from bindings.mapping import ArchProperties
from .problem import Workload
from .model import ArchSpecs


class ArchConstraints(bindings.mapping.ArchConstraints):
    def __init__(self, arch_prop: ArchProperties, workload: Workload,
                 config):
        super().__init__(arch_prop, workload, config)


class Mapping(bindings.mapping.Mapping):
    def __init__(self, config, arch_specs: ArchSpecs,
                 workload: Workload):
        super().__init__(config, arch_specs, workload)
