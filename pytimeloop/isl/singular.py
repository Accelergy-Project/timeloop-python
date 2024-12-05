import islpy as isl


def get_sum_of_pw_qpolynomial(pw_qp):
    sum = get_value_from_singular_qpolynomial(isl.PwQPolynomial.sum(pw_qp))
    if sum.is_nan():
        return 0
    else:
        return sum.to_python()


def get_value_from_singular_qpolynomial(qp):
    return qp.eval(qp.domain().sample_point())