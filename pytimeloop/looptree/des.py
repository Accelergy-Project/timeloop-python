import islpy as isl

import bindings


class LooptreeOutput:
    def __init__(self):
        self.ops = {}
        self.occupancy = {}
        self.op_occupancy = {}
        self.fills_by_peer = {}
        self.fills_by_parent = {}
        self.temporal_steps = {}
        self.fanout = {}


def deserialize_looptree_output(
    looptree_output: bindings.looptree.LooptreeResult,
    isl_ctx: isl.Context
) -> LooptreeOutput:
    output = LooptreeOutput()

    output.ops = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.ops.items()
    }

    output.occupancy = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.occupancy.items()
    }

    output.fills_by_peer = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.fills_by_peer.items()
    }

    output.fills_by_parent = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.fills_by_parent.items()
    }

    output.temporal_steps = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.temporal_steps.items()
    }

    return output