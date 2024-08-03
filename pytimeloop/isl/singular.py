from pytimeloop.isl.top import isl
from pytimeloop.isl.to_str import isl_pw_qpolynomial_to_str


def get_value_from_singular_qpolynomial(qp):
    val_str = isl_pw_qpolynomial_to_str(
        isl.isl_pw_qpolynomial_copy(qp)
    ).strip('{').strip('}').strip()

    val_list = val_str.split('/')
    if len(val_list) == 1:
        return int(val_list[0])
    else:
        return val_list[0]/val_list[1]
