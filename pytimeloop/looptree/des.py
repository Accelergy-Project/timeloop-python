import islpy as isl

import bindings


class LooptreeOutput:
    def __init__(self):
        self.ops = {}
        self.occupancy = {}
        self.fill = {}


def deserialize_looptree_output(
    looptree_output: bindings.looptree.LooptreeResult,
    isl_ctx: isl.Context
) -> LooptreeOutput:
    output = LooptreeOutput()

    output.ops = {
        k: isl.PwQPolynomial.read_from_str(isl_ctx, v)
        for k, v in looptree_output.ops.items()
    }

    output.occupancy = {
        k: isl.PwQPolynomial.read_from_str(isl_ctx, v)
        for k, v in looptree_output.occupancy.items()
    }

    output.fill = {
        k: isl.PwQPolynomial.read_from_str(isl_ctx, v)
        for k, v in looptree_output.fill.items()
    }

    return output