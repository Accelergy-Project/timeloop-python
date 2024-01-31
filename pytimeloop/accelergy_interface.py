from typing import List

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class AccelergyInvocationResult:
    def __init__(self,
                 art_str: str,
                 ert_str: str,
                 stdout_msg: str,
                 stderr_msg: str,
                 art_verbose: str = '',
                 ert_verbose: str = ''):
        self.art = art_str
        self.art_verbose = art_verbose
        self.ert = ert_str
        self.ert_verbose = ert_verbose
        self.stdout_msg = stdout_msg
        self.stderr_msg = stderr_msg

def invoke_accelergy(input_files: List[str], out_dir: str):
    cmd = ['accelergy', *input_files, '-o', out_dir + '/', '-v']
    logger.info(f'Running Accelergy with command: {" ".join(cmd)}')
    result = subprocess.run(cmd,
                            env=os.environ,
                            capture_output=True)
    from pytimeloop.app.call_utils import read_output_files
    ert_str, ert_verbose, art_str, art_verbose = read_output_files(
        result, out_dir, 'accelergy', 'ERT.yaml', 'ERT_summary_verbose.yaml',
        'ART.yaml', 'ART_summary_verbose.yaml'
    )

    return AccelergyInvocationResult(
        art_str,
        ert_str,
        result.stdout,
        result.stderr,
        art_verbose,
        ert_verbose
    )
