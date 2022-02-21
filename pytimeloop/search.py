import bindings
from bindings import SearchStatus
from .config import Config
from .mapspace import MapSpace


class SearchAlgorithm(bindings.SearchAlgorithm):
    @staticmethod
    def parse_and_construct(config: Config, mapspace: MapSpace, id: int):
        _, native_config_node = config.get_native()
        return bindings.SearchAlgorithm.parse_and_construct(
            native_config_node, mapspace, id)
