import islpy as isl

from bindings.looptree import PipelineSpatialTag
from pytimeloop.isl.sum import sum_until_idx, make_reduction_map
from pytimeloop.isl.qpolynomial import from_pw_qpolynomial_fold

def process_sequential_latency(top_idx: int, latencies):
    summed_latency = sum(map(lambda pair: pair[1], latencies))
    return sum_until_idx(top_idx, summed_latency)


def process_parallel_latency(top_idx: int, latencies):
    n_dims = isl.isl_pw_qpolynomial_dim(
        latencies[0][1],
        isl.isl_dim_in
    )
    reduce_to_branching_point_map = make_reduction_map(
        isl.isl_pw_qpolynomial_get_domain_space(latencies[0]),
        n_dims-1
    )

    summed_latency = sum(latencies)

    max_per_timestep, is_tight = \
        reduce_to_branching_point_map.apply_pw_qpolynomial_fold(
            isl.PwQPolynomialFold.from_pw_qpolynomial(
                isl.fold.max,
                summed_latency
            )
        )

    return sum_until_idx(top_idx, max_per_timestep)


def process_pipeline_latency(top_idx: int, latencies):
    sequential_latency = process_sequential_latency(top_idx, latencies)

    summed_latency = sum(map(lambda pair: pair[1], latencies))

    dim_tags = latencies[0][0]
    for pipeline_idx in range(len(dim_tags)):
        if isinstance(dim_tags[pipeline_idx], PipelineSpatialTag):
            break
    summed_latency = sum_until_idx(pipeline_idx+1, summed_latency)
    dim_tags = dim_tags[:pipeline_idx+1]

    space = summed_latency.get_domain_space()
    hidden_latency_map = make_hidden_latency_map(dim_tags,
                                                 space,
                                                 len(latencies))
    hidden_latency_map = \
        hidden_latency_map.intersect_domain(summed_latency.domain())
    hidden_latency_map = \
        hidden_latency_map.intersect_range(summed_latency.domain())
    hidden_latencies = \
        hidden_latency_map.apply_pw_qpolynomial(summed_latency)

    reduction_map = make_reduction_map(space, len(dim_tags)-1, 1)
    reduction_map = reduction_map.intersect_range(summed_latency.domain()).coalesce()
    hidden_latencies, is_tight = \
        reduction_map.apply_pw_qpolynomial_fold(
            isl.PwQPolynomialFold.from_pw_qpolynomial(
                isl.fold.min,
                hidden_latencies
            )
        )
    hidden_latencies = from_pw_qpolynomial_fold(hidden_latencies)

    # Remove last one
    domain = hidden_latencies.domain()
    hidden_latencies = hidden_latencies.subtract_domain(domain.lexmax())

    hidden_latency = sum_until_idx(top_idx, hidden_latencies)

    return sequential_latency - hidden_latency


LATENCY_PROCESSORS = {
    'sequential': process_sequential_latency,
    'parallel': process_parallel_latency,
    'pipeline': process_pipeline_latency
}


def make_hidden_latency_map(dim_tags, space, n_stages):
    """
    space: [..., t, ps]
    returns: [..., t, ps] -> [..., t', ps'] : P*t+ps+1 <= P*t'+ps' < P*t+ps+P
    """
    assert len(dim_tags) >= 2

    t_idx = len(dim_tags)-2
    ps_idx = len(dim_tags)-1

    tprime = isl.Aff.var_on_domain(space, isl.dim_type.set, t_idx)
    ps_prime = isl.Aff.var_on_domain(space, isl.dim_type.set, ps_idx)
    inner = n_stages*tprime + ps_prime

    lower = n_stages*tprime + ps_prime + 1
    upper = n_stages*tprime + ps_prime + n_stages

    return lower.le_map(inner).intersect(upper.gt_map(inner))
