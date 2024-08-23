import ctypes

from pytimeloop.isl.top import isl
from pytimeloop.isl.sum import sum_until_idx, sum_all_pw_qpolynomials, make_reduction_map


def process_sequential_latency(top_idx: int, latencies):
    summed_latency = sum_all_pw_qpolynomials(latencies)
    return sum_until_idx(top_idx, summed_latency)


def process_parallel_latency(top_idx: int, latencies):
    n_dims = isl.isl_pw_qpolynomial_dim(
        latencies[0],
        isl.isl_dim_in
    )
    reduce_to_branching_point_map = make_reduction_map(
        isl.isl_pw_qpolynomial_get_domain_space(latencies[0]),
        n_dims-1
    )

    summed_latency = sum_all_pw_qpolynomials(latencies)

    is_tight = ctypes.c_int(0)
    max_per_timestep = isl.isl_map_apply_pw_qpolynomial_fold(
        reduce_to_branching_point_map,
        isl.isl_pw_qpolynomial_fold_from_pw_qpolynomial(
            isl.isl_fold_max,
            summed_latency
        ),
        ctypes.byref(is_tight)
    )

    return sum_until_idx(top_idx, max_per_timestep)


def process_pipeline_latency(top_idx: int, latencies):
    sequential_latency = process_sequential_latency(top_idx, latencies)

    space = isl.isl_pw_qpolynomial_get_domain_space(latencies[0])
    n_dims = isl.isl_pw_qpolynomial_dim(
        latencies[0],
        isl.isl_dim_in
    )
    hidden_latency_map = make_hidden_latency_map(n_dims, space)

    summed_latency = sum_all_pw_qpolynomials(latencies)

    hidden_latencies = isl.isl_map_apply_pw_qpolynomial(
        hidden_latency_map,
        summed_latency
    )

    is_tight = ctypes.c_int(0)
    hidden_latencies = isl.isl_map_apply_pw_qpolynomial_fold(
        make_reduction_map(n_dims-1, space),
        isl.isl_pw_qpolynomial_fold_from_pw_qpolynomial(
            isl.isl_fold_max,
            hidden_latencies
        ),
        ctypes.byref(is_tight)
    )

    hidden_latency = sum_until_idx(top_idx, hidden_latencies)

    return isl.isl_pw_qpolynomial_sub(sequential_latency,
                                      hidden_latency)


LATENCY_PROCESSORS = {
    'sequential': process_sequential_latency,
    'parallel': process_parallel_latency,
    'pipeline': process_pipeline_latency
}
