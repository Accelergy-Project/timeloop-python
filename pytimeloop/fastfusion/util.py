from numbers import Number

from joblib import Parallel

import sys

from tqdm import tqdm

N_PARALLEL_THREADS = 64


class fzs(frozenset):
    def __repr__(self):
        return f"fzs({', '.join(sorted(x.__repr__() for x in self))})"
    
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


def parallel(jobs, n_jobs: int = None, one_job_if_debugging: bool = True, pbar: str = None, return_as: str = None):
    args = {}
    if return_as is not None:
        args["return_as"] = return_as
        
    if n_jobs is None:
        n_jobs = N_PARALLEL_THREADS

    if one_job_if_debugging and debugger_active():
        n_jobs = 1

    if isinstance(jobs, dict):
        assert return_as == None, "return_as is not supported for dict jobs"
        r = zip(jobs.keys(), parallel(jobs.values(), pbar=pbar, one_job_if_debugging=one_job_if_debugging))
        return {k: v for k, v in r}

    if n_jobs == 1:
        return [j[0](*j[1], **j[2]) for j in jobs]
    
    if pbar:
        return Parallel(n_jobs=n_jobs, **args)(tqdm(jobs, total=len(jobs), desc=pbar, leave=True))
    return Parallel(n_jobs=n_jobs, **args)(jobs)

