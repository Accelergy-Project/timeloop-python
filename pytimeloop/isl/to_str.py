from ctypes import c_char_p

from pytimeloop.isl.top import isl


def make_return_python_str(func):
    def func_returning_python_str(*arg, **kwargs):
        c_res = func(*arg, **kwargs)
        return c_char_p(c_res).value.decode('utf-8')
    return func_returning_python_str


@make_return_python_str
def isl_map_to_str(map):
    return isl.isl_map_to_str(map)


@make_return_python_str
def isl_pw_qpolynomial_to_str(pw_qp):
    return isl.isl_pw_qpolynomial_to_str(pw_qp)
