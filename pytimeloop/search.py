from bindings import NativeSearchAlgorithm, SearchStatus
from .config import Config
from .mapspace import MapSpace


class SearchAlgorithm(NativeSearchAlgorithm):
    @staticmethod
    def parse_and_construct(config: Config, mapspace: MapSpace, id: int):
        _, native_config_node = config.get_native()
        return NativeSearchAlgorithm.parse_and_construct(
            native_config_node, mapspace, id)
