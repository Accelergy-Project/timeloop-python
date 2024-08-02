"""Takes constraints from constraints lists and attaches them to
objects in the architecture.
"""
from .sparse_opt_attacher import SparseOptAttacherProcessor
from pytimeloop.timeloopfe.v4.arch import Compute, Component, Storage
from ...common.processor import References2CopiesProcessor
from ...common.nodes import Node
from ...common.processor import Processor
from ...v4 import Specification


class RequiredActionsProcessor(Processor):
    """Ensures that all components have actions defined for Accelergy
    Storage:
    - read
    - write
    - update
    # - metadata_update

    DEPRECATED. NOW ALL OF THE FOLLOWING ARE NOT REQUIRED.
    Deprecated because: Skipped/gated actions can just be no action
    decompression_count/compression_count is not supported in many of the
    plug-ins that Sparseloop uses
    # Storage if metadata attributes are present:
    # - metadata_read
    # - metadata_write
    # Storage if sparse optimization is enabled:
    # - gated_read
    # - gated_write
    # - gated_update
    # - skipped_read
    # - skipped_write
    # - skipped_update
    # - decompression_count
    # - compression_count
    # Storage if sparse and metadata are enabled:
    # - gated_metadata_read
    # - skipped_metadata_read
    # - gated_metadata_write
    # - skipped_metadata_write
    # - gated_metadata_write
    # - skipped_metadata_write
    # Compute:
    # - compute
    # Compute if sparse optimization is enabled:
    # - gated_compute
    # - skipped_compute
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_storage(self, elem: Storage):
        sparse_opts = elem.sparse_optimizations

        has_metadata = elem.attributes.metadata_datawidth is not None
        sparse_rep = not Node.isempty_recursive(sparse_opts.representation_format)
        sparse_action = not Node.isempty_recursive(sparse_opts.action_optimization)
        read_write_update = [""]
        required_actions = ["leak"]
        # if has_metadata:
        #     read_write_update.append("metadata_")
        # if sparse_action:
        #     read_write_update.append("gated_")
        #     read_write_update.append("skipped_")
        # if sparse_action and has_metadata:
        #     read_write_update.append("gated_metadata_")
        #     read_write_update.append("skipped_metadata_")
        # if sparse_rep:
        #     required_actions += ["decompression_count", "compression_count"]
        for r in read_write_update:
            for a in ["read", "write", "update"]:
                required_actions.append(r + a)
        elem.required_actions = list(set(required_actions + elem.required_actions))

    def check_compute(self, elem: Component):
        sparse_opts = elem.sparse_optimizations
        sparse_action = not Node.isempty(sparse_opts.action_optimization)
        required_actions = ["compute"]
        # if sparse_action:
        #     required_actions += ["gated_compute", "skipped_compute"]
        elem.required_actions = list(set(required_actions + elem.required_actions))

    def process(self, spec: Specification):
        super().process(spec)
        self.must_run_after(References2CopiesProcessor, spec)
        self.must_run_after(SparseOptAttacherProcessor, spec)
        for s in spec.architecture.get_nodes_of_type(Storage):
            self.check_storage(s)  # type: ignore
        for c in spec.architecture.get_nodes_of_type(Compute):
            self.check_compute(c)  # type: ignore
