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


def invoke_accelergy(input_files: List[str], out_prefix: str,
                     out_dir: str):
    result = subprocess.run(['accelergy', *input_files,
                             '--oprefix', out_prefix + '.',
                             '-o', out_dir + '/',
                             '-v 1'],
                            capture_output=True)

    PATH_TO_ERT = os.path.join(out_dir, out_prefix + '.ERT.yaml')
    PATH_TO_ERT_VERBOSE = os.path.join(out_dir,
                                       out_prefix + '.ERT_summary_verbose.yaml')
    PATH_TO_ART = os.path.join(out_dir, out_prefix + '.ART.yaml')
    PATH_TO_ART_VERBOSE = os.path.join(out_dir,
                                       out_prefix + '.ART_summary_verbose.yaml')

    ert_str = ''
    if os.path.isfile(PATH_TO_ERT):
        with open(PATH_TO_ERT, 'r') as f:
            ert_str = f.read()
        os.remove(PATH_TO_ERT)
    else:
        logger.error('Could not find ERT')

    ert_verbose = ''
    if os.path.isfile(PATH_TO_ERT_VERBOSE):
        with open(PATH_TO_ERT_VERBOSE, 'r') as f:
            ert_verbose = f.read()
        os.remove(PATH_TO_ERT_VERBOSE)
    else:
        logger.warn('Could not find ERT_verbose')

    art_str = ''
    if os.path.isfile(PATH_TO_ART):
        with open(PATH_TO_ART, 'r') as f:
            art_str = f.read()
        os.remove(PATH_TO_ART)
    else:
        logger.error('Could not find ART')

    art_verbose = ''
    if os.path.isfile(PATH_TO_ART_VERBOSE):
        with open(PATH_TO_ART_VERBOSE, 'r') as f:
            art_verbose = f.read()
        os.remove(PATH_TO_ART_VERBOSE)
    else:
        logger.warn('Could not find ART_verbose')

    return AccelergyInvocationResult(
        art_str,
        ert_str,
        result.stdout,
        result.stderr,
        art_verbose,
        ert_verbose
    )
