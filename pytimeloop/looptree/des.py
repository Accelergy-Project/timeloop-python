from ctypes import c_char_p

from pytimeloop.isl.top import Context, isl, libc

import bindings


class LooptreeOutput:
    def __init__(self):
        self.ops = {}
        self.occupancy = {}
        self.fill = {}


def deserialize_looptree_output(
    looptree_output: bindings.looptree.LooptreeResult,
    isl_ctx: Context
) -> LooptreeOutput:
    output = LooptreeOutput()

    output.ops = {
        k: isl.isl_pw_qpolynomial_read_from_str(
            isl_ctx.from_param(),
            c_char_p(v.encode('utf-8'))
        )
        for k, v in looptree_output.ops.items()
    }

    output.occupancy = {
        k: isl.isl_pw_qpolynomial_read_from_str(
            isl_ctx.from_param(),
            c_char_p(v.encode('utf-8'))
        )
        for k, v in looptree_output.occupancy.items()
    }

    output.fill = {
        k: isl.isl_pw_qpolynomial_read_from_str(
            isl_ctx.from_param(),
            c_char_p(v.encode('utf-8'))
        )
        for k, v in looptree_output.fill.items()
    }

    return output