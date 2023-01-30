import bindings
from bindings.search import SearchStatus
from .mapspace import MapSpace


class SearchAlgorithm(bindings.search.SearchAlgorithm):
    @staticmethod
    def parse_and_construct(config, mapspace: MapSpace, id: int):
        _, native_config_node = config.get_native()
        return bindings.search.SearchAlgorithm.parse_and_construct(
            native_config_node, mapspace, id)
