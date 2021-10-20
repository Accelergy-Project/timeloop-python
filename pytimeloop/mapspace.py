from bindings import NativeMapSpace, Dimension, ID
from .config import Config
from .model import ArchSpecs
from .problem import Workload


class MapSpace(NativeMapSpace):
    @staticmethod
    def parse_and_construct(config: Config, arch_constraints: Config,
                            arch_specs: ArchSpecs, workload: Workload):
        _, native_config_node = config.get_native()
        _, native_arch_const_node = arch_constraints.get_native()

        return NativeMapSpace.parse_and_construct(native_config_node,
                                                  native_arch_const_node,
                                                  arch_specs,
                                                  workload)
