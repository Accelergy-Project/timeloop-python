from collections.abc import Sequence
import islpy as isl

from .reduction import make_reduction_map, make_reduction_map_from_mask


def sum_until_idx(n_dims_left: int, pw_qp):
    n_dims_in = pw_qp.dim(isl.dim_type.in_)
    dims_out_first = n_dims_left
    n_dims_out = n_dims_in - n_dims_left
    reduction_map = make_reduction_map(pw_qp.get_domain_space(),
                                       dims_out_first,
                                       n_dims_out)
    return reduction_map.apply_pw_qpolynomial(pw_qp)


def sum_with_mask(mask: Sequence[bool], pw_qp):
    reduction_map = make_reduction_map_from_mask(pw_qp.get_domain_space(),
                                                 mask)
    return reduction_map.apply_pw_qpolynomial(pw_qp)
