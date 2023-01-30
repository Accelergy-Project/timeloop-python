from typing import List
from bindings.accelergy import native_invoke_accelergy

import os
import subprocess

class AccelergyInvocationResult:
    def __init__(self,
                 art_str: str,
                 ert_str: str,
                 stdout_msg: str,
                 stderr_msg: str):
        self.art = art_str
        self.ert = ert_str
        self.stdout_msg = stdout_msg
        self.stderr_msg = stderr_msg


def invoke_accelergy(input_files: List[str], out_prefix: str,
                     out_dir: str):
    result = subprocess.run(['accelergy', *input_files,
                             '-f', 'ART', 'ERT',
                             '--oprefix', out_prefix + '.',
                             '-o', out_dir + '/'],
                            capture_output=True)

    path_to_ert = os.path.join(out_dir, out_prefix + '.ERT.yaml')
    ert_str = None
    with open(path_to_ert, 'r') as f:
        ert_str = f.read()
    os.remove(path_to_ert)

    path_to_art = os.path.join(out_dir, out_prefix + '.ART.yaml')
    art_str = None
    with open(path_to_art, 'r') as f:
        art_str = f.read()
    os.remove(path_to_art)

    return AccelergyInvocationResult(art_str, ert_str,
                                     result.stdout, result.stderr)
