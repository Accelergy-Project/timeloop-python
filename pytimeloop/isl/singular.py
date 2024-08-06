import islpy as isl


def get_sum_of_pw_qpolynomial(pw_qp):
    return get_value_from_singular_qpolynomial(isl.PwQPolynomial.sum(pw_qp))


def get_value_from_singular_qpolynomial(qp):
    return qp.eval(qp.domain().sample_point())