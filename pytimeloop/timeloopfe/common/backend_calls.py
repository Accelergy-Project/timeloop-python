"""Call Timeloop from Python"""

import copy
import os
import signal
import subprocess
import sys
from typing import Any, List, Optional, Dict, Tuple, Union
import logging
from accelergy.utils.yaml import to_yaml_string
import psutil
from .base_specification import BaseSpecification

DELAYED_IMPORT_DONE = False


def delayed_import():
    global DELAYED_IMPORT_DONE
    if DELAYED_IMPORT_DONE:
        return
    global v3spec, v4spec, v4_to_v3, v4fusedspec

    from ..v3 import specification as current_import

    v3spec = current_import
    from ..v4 import specification as current_import

    v4spec = current_import
    from .version_transpilers import v4_to_v3 as current_import

    v4_to_v3 = current_import

    from ..v4fused import specification as current_import
    v4fusedspec = current_import


def _specification_to_yaml_string(
    specification: BaseSpecification,
    for_model: bool = False,
) -> str:
    """Converts specification into YAML string, which may require transpilation.
    !@param specification The specification with which to call Timeloop.
    !@param for_model Whether the result is for Timeloop model or mapper
    """
    delayed_import()
    specification = specification._process()
    if specification.processors and not specification._processors_run:
        raise RuntimeError(
            "Specification has not been processed yet. Please call "
            "spec.process() before calling Timeloop or Accelergy."
        )

    if isinstance(specification, v3spec.Specification):
        input_content = to_yaml_string(specification)
    elif isinstance(specification, v4spec.Specification):
        input_content = v4_to_v3.transpile(specification, for_model=for_model)
        input_content = to_yaml_string(input_content)
    elif isinstance(specification, v4fusedspec.Specification):
        input_content = v4_to_v3.transpile(specification, add_spatial_dummy=False)
        input_content = to_yaml_string(input_content)
    else:
        raise TypeError(f"Can not call Timeloop with {type(specification)}")

    return input_content


def _pre_call(
    specification: BaseSpecification,
    output_dir: str,
    extra_input_files: Optional[List[str]] = None,
    for_model: bool = False,
) -> Tuple[List[str], str]:
    """Prepare to call Timeloop or Accelergy from Python
    !@param specification The specification with which to call Timeloop.
    !@param output_dir The directory to run Timeloop in.
    !@param extra_input_files A list of extra input files to pass to Timeloop.
    !@param for_model Whether the result is for Timeloop model or mapper
    """
    delayed_import()

    input_content = _specification_to_yaml_string(specification, for_model)

    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "parsed-processed-input.yaml"),
        "w",
    ) as f:
        f.write(input_content)

    input_paths = [os.path.join(output_dir, "parsed-processed-input.yaml")] + (
        extra_input_files or []
    )
    input_paths = [os.path.abspath(f) for f in input_paths]
    return (
        input_paths,
        output_dir,
    )


def _call(
    call: str,
    input_paths: List[str],
    output_dir: str,
    environment: Optional[Dict[str, str]] = None,
    dump_intermediate_to: Optional[str] = None,
    log_to: Optional[str] = None,
    extra_args: List[str] = (),
    return_proc: bool = False,
) -> Union[int, subprocess.Popen]:
    """Call a Timeloop or Accelergy command from Python

    Args:
        call (str): Which command to call.
        input_paths (List[str]): The content of the input file.
        output_dir (str): The directory to run Timeloop in.
        environment (Optional[Dict[str, str]]): A dictionary of environment variables to pass.
        dump_intermediate_to (Optional[str]): If not None, dump the input content to this file before calling.
        log_to (Optional[str]): If not None, log the output of the call to this file or file-like object.
        extra_args (List[str]): A list of extra arguments to pass to the call.
        return_proc (bool): If True, return the subprocess.Popen object instead of the return code.

    Returns:
        Union[int, subprocess.Popen]: The return code of the call, or the subprocess.Popen object if return_proc is True.
    """
    os.makedirs(output_dir, exist_ok=True)
    if dump_intermediate_to is None:
        dump_intermediate_to = os.path.join(
            output_dir, f"tl-parsed-processed-input.yaml"
        )
    logging.info(f"Calling {call} with input {input_paths} and output {output_dir}")

    ifiles = [os.path.abspath(x) for x in input_paths]
    for i, f in enumerate(ifiles):
        # If not quote enclosed, add quotes
        if not f.startswith('"') and not f.endswith('"'):
            ifiles[i] = '"' + f + '"'

    envstr = " ".join(str(k) + "=" + str(v) for k, v in (environment or {}).items())

    tlcmd = (
        f'cd "{output_dir}" ; {envstr} {call} '
        f'{" ".join(extra_args)} {" ".join(ifiles)}'
    )
    logging.info("Calling Timeloop: %s", tlcmd)
    if isinstance(log_to, str):
        log_to = open(log_to, "w")
    if log_to is None:
        # Send to the current stdout
        log_to = sys.stdout
    proc = subprocess.Popen(tlcmd, shell=True, stdout=log_to, stderr=subprocess.STDOUT)
    if return_proc:
        return proc
    else:
        # Wait for the subprocess. If there is a ctrl-c, send it to the proc.
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.send_signal(sig=signal.SIGINT)


def _parse_output(
    specification: BaseSpecification,
    output_dir: str,
    result: Union[int, subprocess.Popen],
    for_model: bool = False,
) -> Union[int, subprocess.Popen]:
    """ """
    if isinstance(result, subprocess.Popen):
        return result

    m = "model" if for_model else "mapper"
    errmsg = (
        f"\n\n" + "=" * 120 + "\n"
        f"Timeloop {m} failed with return code {result}. Please check the output files "
        f"in {output_dir} for more information. To debug, you can edit the file:"
        f"\n\t{os.path.join(output_dir, 'parsed-processed-input.yaml')}\nand run "
        f"\n\ttl {m} {os.path.join(output_dir, 'parsed-processed-input.yaml')}\nto see the error. "
        f"If you're running the mapper and Timeloop can't find a vaild mapping, try setting "
        f"'diagnostics: true' in the mapper input specification."
    )

    if result != 0:
        raise RuntimeError(errmsg)
    try:
        return specification._parse_timeloop_output(
            output_dir, prefix="timeloop-model" if for_model else "timeloop-mapper"
        )
    except Exception as e:
        raise RuntimeError(f"{errmsg}") from e


def call_mapper(
    specification: BaseSpecification,
    output_dir: str,
    environment: Optional[Dict[str, str]] = None,
    extra_input_files: Optional[List[str]] = None,
    dump_intermediate_to: Optional[str] = None,
    log_to: Optional[Union[str, Any]] = None,
    extra_args: List[str] = (),
    return_proc: bool = False,
) -> Union[int, subprocess.Popen]:
    """Call Timeloop Mapper from Python

    Args:
        specification (BaseSpecification): The specification with which to call Timeloop.
        output_dir (str): The directory to run Timeloop in.
        environment (Optional[Dict[str, str]]): A dictionary of environment variables to pass to Timeloop.
        extra_input_files (Optional[List[str]]): A list of extra input files to pass to Timeloop
        dump_intermediate_to (Optional[str]): If not None, dump the input content to this file before calling Timeloop.
        log_to (Optional[Union[str, Any]]): If not None, log the output of the Timeloop call to this file or file-like object.
        extra_args (List[str]): A list of extra arguments to pass to Timeloop.
        return_proc (bool): If True, return the subprocess.Popen object instead of the return code.

    Returns:
        Union[int, subprocess.Popen]: The return code of the call, or the subprocess.Popen object if return_proc is True.
    """
    input_paths, output_dir = _pre_call(
        specification, output_dir, extra_input_files, for_model=False
    )

    return _parse_output(
        specification=specification,
        output_dir=output_dir,
        result=_call(
            "timeloop-mapper",
            input_paths=input_paths,
            output_dir=output_dir,
            environment=environment,
            dump_intermediate_to=dump_intermediate_to,
            log_to=log_to,
            extra_args=extra_args,
            return_proc=return_proc,
        ),
        for_model=False,
    )


def call_model(
    specification: BaseSpecification,
    output_dir: str,
    environment: Optional[Dict[str, str]] = None,
    extra_input_files: Optional[List[str]] = None,
    dump_intermediate_to: Optional[str] = None,
    log_to: Optional[Union[str, Any]] = None,
    extra_args: List[str] = (),
    return_proc: bool = False,
) -> Union[int, subprocess.Popen]:
    """Call Timeloop Model from Python

    Args:
        specification (BaseSpecification): The specification with which to call Timeloop.
        output_dir (str): The directory to run Timeloop in.
        environment (Optional[Dict[str, str]]): A dictionary of environment variables to pass to Timeloop
        extra_input_files (Optional[List[str]]): A list of extra input files to pass to Timeloop
        dump_intermediate_to (Optional[str]): If not None, dump the input content to this file before calling Timeloop.
        log_to (Optional[Union[str, Any]]): If not None, log the output of the Timeloop call to this file or file-like object.
        extra_args (List[str]): A list of extra arguments to pass to Timeloop.
        return_proc (bool): If True, return the subprocess.Popen object instead of the return code.

    Returns:
        Union[int, subprocess.Popen]: The return code of the call, or the subprocess.Popen object if return_proc is True.
    """
    input_paths, output_dir = _pre_call(
        specification, output_dir, extra_input_files, for_model=True
    )

    return _parse_output(
        specification=specification,
        output_dir=output_dir,
        result=_call(
            "timeloop-model",
            input_paths=input_paths,
            output_dir=output_dir,
            environment=environment,
            dump_intermediate_to=dump_intermediate_to,
            log_to=log_to,
            extra_args=extra_args,
            return_proc=return_proc,
        ),
        for_model=True,
    )


def call_accelergy_verbose(
    specification: BaseSpecification,
    output_dir: str,
    environment: Optional[Dict[str, str]] = None,
    extra_input_files: Optional[List[str]] = None,
    dump_intermediate_to: Optional[str] = None,
    log_to: Optional[Union[str, Any]] = None,
    extra_args: List[str] = (),
    return_proc: bool = False,
) -> Union[int, subprocess.Popen]:
    """Call Accelergy from Python

    Args:
        specification (BaseSpecification): The specification with which to call Accelergy.
        output_dir (str): The directory to run Accelergy in.
        environment (Optional[Dict[str, str]]): A dictionary of environment variables to pass to Accelergy.
        extra_input_files (Optional[List[str]]): A list of extra input files to pass to Accelergy
        dump_intermediate_to (Optional[str]): If not None, dump the input content to this file before calling Accelergy.
        log_to (Optional[Union[str, Any]]): If not None, log the output of the Accelergy call to this file or file-like object.
        extra_args (List[str]): A list of extra arguments to pass to Accelergy.
        return_proc (bool): If True, return the subprocess.Popen object instead of the return code.
    """
    input_paths, output_dir = _pre_call(
        specification, output_dir, extra_input_files, for_model=False
    )

    return _call(
        "accelergy -v",
        input_paths=input_paths,
        output_dir=output_dir,
        environment=environment,
        dump_intermediate_to=dump_intermediate_to,
        log_to=log_to,
        extra_args=extra_args,
        return_proc=return_proc,
    )


def call_stop(
    proc: Optional[subprocess.Popen] = None,
    max_wait_time: Optional[int] = None,
    force: bool = False,
):
    """Stop Timeloop subprocesses.

    Args:
        proc (Optional[subprocess.Popen]): The subprocess to stop. If None, stop all Timelojson processes.
        max_wait_time (Optional[int]): The maximum time to wait for the process to stop. If 0, do not wait.
        force (bool): If True, force kill the process instead of sending SIGINT.
    """

    def stop_single(p, f):
        if f:
            logging.info("  Force killing process PID %s", p.pid)
            p.kill()
        else:
            logging.info("  Sending SIGINT to process PID %s", p.pid)
            p.send_signal(signal.SIGINT)

    def stop_proc(p, f):
        logging.info("Stopping %s", p.pid)
        children = psutil.Process(p.pid).children(recursive=True)
        for child in children:
            stop_single(child, f)
        stop_single(p, f)

    procs = []
    if proc is None:
        procs = [p for p in psutil.process_iter() if "timeloop" in p.name()]
        procs += [p for p in psutil.process_iter() if "accelergy" in p.name()]
    else:
        procs = [proc]

    for p in procs:
        try:
            stop_proc(p, force)
        except psutil.NoSuchProcess:
            pass
    if max_wait_time is not None:
        for p in procs:
            try:
                p.wait(None if max_wait_time == 0 else max_wait_time)
            except psutil.NoSuchProcess:
                pass


def accelergy_app(
    specification: BaseSpecification,
    output_dir: str,
    extra_input_files: Optional[List[str]] = None,
) -> "AccelergyInvocationResult":
    """Call the PyTimeloop Accelergy interface

    Args:
        specification (BaseSpecification): The specification with which to call Accelergy.
        output_dir (str): The directory to run Accelergy in.
        extra_input_files (Optional[List[str]]): A list of extra input files to pass to Accelergy

    Returns:
        AccelergyInvocationResult: The result of the Accelergy invocation.
    """
    try:
        from pytimeloop.accelergy_interface import invoke_accelergy
    except:
        raise ImportError(
            "pytimeloop is not installed. To call accelergy_app, please install pytimeloop. "
            "Alternatively, you can use the call_accelergy_verbose function directly."
        )

    input_paths, output_dir = _pre_call(specification, output_dir, extra_input_files)
    return invoke_accelergy(input_paths, output_dir)


def to_mapper_app(
    specification: BaseSpecification,
    output_dir: str,
    extra_input_files: Optional[List[str]] = None,
):
    """
    Create a PyTimeloop MapperApp object from a specification.

    Args:
        specification (BaseSpecification): The specification with which to call Timeloop.
        output_dir (str): The directory to run Timeloop in.
        extra_input_files (Optional[List[str]]): A list of extra input files to pass to Timeloop

    Returns:
        MapperApp: The MapperApp object.
    """
    try:
        from pytimeloop.app import MapperApp
        from pytimeloop.config import Config
    except ImportError:
        raise ImportError(
            "pytimeloop is not installed. To create a mapper app, please install pytimeloop. "
            "Alternatively, you can use the call_mapper function directly."
        )
    input_content = _specification_to_yaml_string(specification,
                                                  for_model=False)

    if extra_input_files is not None:
        for fname in extra_input_files:
            with open(fname, 'r') as f:
                input_content += '\n'
                input_content += f.read()

    config = Config(input_content, 'yaml')
    return MapperApp(config, output_dir, 'timeloop-mapper')


def to_model_app(
    specification: BaseSpecification,
    output_dir: str,
    extra_input_files: Optional[List[str]] = None,
):
    """
    Create a PyTimeloop ModelApp object from a specification.

    Args:
        specification (BaseSpecification): The specification with which to call Timeloop.
        output_dir (str): The directory to run Timeloop in.
        extra_input_files (Optional[List[str]]): A list of extra input files to pass to Timeloop

    Returns:
        ModelApp: The ModelApp object.
    """
    try:
        from pytimeloop.app import ModelApp
        from pytimeloop.config import Config
    except ImportError:
        raise ImportError(
            "pytimeloop is not installed. To create a model app, please install pytimeloop. "
            "Alternatively, you can use the call_model function directly."
        )
    input_content = _specification_to_yaml_string(specification,
                                                  for_model=True)

    if extra_input_files is not None:
        for fname in extra_input_files:
            with open(fname, 'r') as f:
                input_content += '\n'
                input_content += f.read()

    config = Config(input_content, 'yaml')
    return ModelApp(config, output_dir, 'timeloop-model')
