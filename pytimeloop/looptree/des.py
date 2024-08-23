import islpy as isl

import bindings


class LooptreeOutput:
    def __init__(self):
        self.ops = {}
        self.occupancy = {}
        self.op_occupancy = {}
        self.fill = {}


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
        k: isl.PwQPolynomial.read_from_str(isl_ctx, v)
        for k, v in looptree_output.occupancy.items()
    }

    output.fill = {
        k: isl.PwQPolynomial.read_from_str(isl_ctx, v)
        for k, v in looptree_output.fill.items()
    }

    output.temporal_steps = {
        k: (dims, isl.PwQPolynomial.read_from_str(isl_ctx, v))
        for k, (dims, v) in looptree_output.temporal_steps.items()
    }

    return output