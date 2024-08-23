import islpy as isl


def from_qpolynomial_fold(qp_fold):
    qp = None
    def gather(qp_):
        nonlocal qp
        qp = qp_
    stat = qp_fold.foreach_qpolynomial(gather)
    return qp


def from_pw_qpolynomial_fold(pw_qp_fold):
    pw_qp = None
    def gather(set, qp_fold):
        nonlocal pw_qp
        qp = isl.PwQPolynomial.from_qpolynomial(
            from_qpolynomial_fold(qp_fold)
        )
        if pw_qp is None:
            pw_qp = qp.intersect_domain(set)
        else:
            pw_qp += qp.intersect_domain(set)
    pw_qp_fold.foreach_piece(gather)
    return pw_qp