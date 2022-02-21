import bindings
from .config import Config


class Workload(bindings.Workload):
    def __init__(self, config: Config):
        _, native_workload_cfg_node = config.get_native()
        super().__init__(native_workload_cfg_node)
