import bindings
from .config import Config


class Workload(bindings.problem.Workload):
    def __init__(self, config):
        super().__init__(config)
