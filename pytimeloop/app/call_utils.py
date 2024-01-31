import os
import subprocess


def read_output_files(
    subproc: subprocess.CompletedProcess, out_dir: str, name: str, *files: str
):
    stdout = os.path.join(out_dir, name + ".stdout.log")
    stderr = os.path.join(out_dir, name + ".stderr.log")
    failstr = f"{name} failed to run. Check {stdout} and {stderr} for more details."
    for to_write, path in ((stdout, subproc.stdout), (stderr, subproc.stderr)):
        with open(to_write, "w") as f:
            f.write(path.decode("utf-8"))
    if subproc.returncode != 0:
        raise Exception(failstr)
    rval = []
    for file in files:
        path = os.path.join(out_dir, file)
        if os.path.isfile(path):
            with open(path, "r") as f:
                rval.append(f.read())
        else:
            raise Exception(f"Could not find {path}. {failstr}")
    return rval
