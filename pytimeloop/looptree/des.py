import islpy as isl

import bindings


class LooptreeOutput:
    def __init__(self):
        self.ops = {}
        self.fills = {}
        self.occupancy = {}
        self.op_occupancy = {}
        self.reads_to_peer = {}
        self.reads_to_parent = {}
        self.temporal_steps = {}
        self.fanout = {}

    def __repr__(self):
        return (
            f'LooptreeOutput(' +
            f'ops={self.ops}, ' +
            f'occupancy={self.occupancy}, ' +
            f'reads_to_parent={self.reads_to_parent})'
        )


def deserialize_looptree_output(
    looptree_output: bindings.looptree.LooptreeResult,
    isl_ctx: isl.Context
) -> LooptreeOutput:
    output = LooptreeOutput()

    output.ops = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.ops.items()
    }

    output.fills = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.fills.items()
    }

    output.occupancy = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.occupancy.items()
    }

    output.reads_to_peer = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.reads_to_peer.items()
    }

    output.reads_to_parent = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.reads_to_parent.items()
    }

    output.temporal_steps = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.temporal_steps.items()
    }

    return output