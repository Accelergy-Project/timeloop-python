"""Enables the dummy table for Accelergy to placeholder energy/area."""
from ..arch import Component
from ...common.processor import Processor
from ...v4 import Specification


class EnableDummyTableProcessor(Processor):
    """
    Enable the dummy table for Accelergy to placeholder energy/area.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, spec: Specification):
        super().process(spec)
        for c in spec.get_nodes_of_type((Component)):
            c.required_actions.extend(["read", "write", "update", "leak"])
            c.attributes["technology"] = -1
