import bindings
from bindings.mapping import ArchProperties
from .config import Config
from .problem import Workload
from .model import ArchSpecs


class ArchConstraints(bindings.mapping.ArchConstraints):
    def __init__(self, arch_prop: ArchProperties, workload: Workload,
                 config: Config):
        _, native_config_node = config.get_native()
        super().__init__(arch_prop, workload, native_config_node)


class Mapping(bindings.mapping.Mapping):
    def __init__(self, config: Config, arch_specs: ArchSpecs,
                 workload: Workload):
        _, workload_config_node = config.get_native()
        super().__init__(workload_config_node, arch_specs, workload)
