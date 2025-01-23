from numbers import Number

from joblib import Parallel

import sys

N_PARALLEL_THREADS = 128


class fzs(frozenset):
    def __repr__(self):
        try:
            return f"fzs({', '.join(x.__repr__() for x in sorted(self))}"
        except:
            return f"fzs({', '.join(x.__repr() for x in self)})"
    
    def __str__(self):
        return self.__repr__()

def debugger_active():
    return sys.gettrace() is not None


def expfmt(x):
    if isinstance(x, Number):
        x = round(x)
        if x < 10000:
            return f"{x}"
        x = f"{x:.2e}"
    else:
        x = str(x)
    if "e+00" in x:
        x = x.replace("e+00", "")
    x = x.replace("e+", "e")
    return x


def parallel(jobs, n_jobs: int = None, one_job_if_debugging: bool = True):
    if n_jobs is None:
        n_jobs = N_PARALLEL_THREADS

    if one_job_if_debugging and debugger_active():
        n_jobs = 1

    if isinstance(jobs, dict):
        return {k: v for k, v in zip(jobs.keys(), parallel(jobs.values()))}

    if n_jobs == 1:
        return [j[0](*j[1], **j[2]) for j in jobs]

    return Parallel(n_jobs=n_jobs)(jobs)
