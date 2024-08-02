import argparse
import glob
import os
import re
from typing import List

SUPPORTED_VERSIONS = ["0.3", "0.4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
    Timeloop version 0.4
    Documentation: https://timeloop.csail.mit.edu/v4
    Examples: https://github.com/Accelergy-Project/timeloop-accelergy-exercises
    """
    )

    parser.add_argument(
        "app", choices=["model", "mapper", "accelergy"], help="Application to run"
    )
    parser.add_argument(
        "-a",
        choices=["model", "mapper", "accelergy"],
        help="Additional application to run",
    )

    parser.add_argument("input_files", nargs="*", help="Input files")
    parser.add_argument("-o", "--output_dir", help="Output directory", default=".")
    parser.add_argument(
        "-j",
        "--jinja_parse_data",
        nargs="+",
        help="Dictionary of data to parse with Jinja2 templating",
    )
    parser.add_argument(
        "-l",
        "--list-components",
        action="store_true",
        default=False,
        help="List components and actions supported by Accelergy.",
    )

    return parser.parse_args()


def get_version(input_files: List[str]) -> object:
    versions = set()
    for file in input_files:
        found = glob.glob(file)
        if not found:
            raise FileNotFoundError(f"File {file} not found")
        for file in glob.glob(file):
            with open(file, "r") as f:
                content = f.read()
                match = re.search(r"version:[\s'\"]*(\d+\.\d+)", content)
                if match:
                    versions.add(match.group(1))
                if re.search(r"dumped_by_timeloop_front_end", content):
                    return None

    if not versions:
        raise ValueError(
            'No version found in input files. Please have a "version: X.Y" '
            "somewhere in the input files, where X and Y are integers."
        )

    assert (
        len(versions) == 1
    ), "All input files must have the same version. Found text " + ", ".join(
        f'"version: {v}"' for v in versions
    )

    version = versions.pop()
    assert version in SUPPORTED_VERSIONS, (
        f"Version {version} not supported. Supported "
        f"versions: {', '.join(SUPPORTED_VERSIONS)}"
    )

    if version == "0.4":
        import pytimeloop.timeloopfe.v4 as tl
    elif version == "0.3":
        import pytimeloop.timeloopfe.v3 as tl
    else:
        raise ValueError(f"Version {version} not supported. V")

    print(f"Running Timeloop version {version}")

    return tl


def call_no_parse(apps: List[str], args: argparse.Namespace, tl: object):
    print(
        "Found parsed-processed-input.yaml in input files. "
        "Running Timeloop without parsing or processing steps. "
        "If this is not the intended behavior, please name the "
        "input files differently."
    )
    assert (
        not args.jinja_parse_data
    ), "Cannot use jinja parse data with parsed-processed-input.yaml"

    for app in apps:
        if app in ["model", "mapper"]:
            app = f"timeloop-{app}"
        elif app == "accelergy":
            app = "accelergy -v"
        extra_args = ["-l"] if args.list_components else []
        tl.backend_calls._call(
            app, args.input_files, output_dir=args.output_dir, extra_args=extra_args
        )


def get_jinja_parse_data(args: argparse.Namespace) -> dict:
    jinja_parse_data = {}
    for data in args.jinja_parse_data or []:
        if "=" not in data:
            raise ValueError(
                f"Invalid jinja parse data: {data}. Must be in the form 'key=value'"
            )
        key, value = data.split("=")
        if not key or not value:
            raise ValueError(
                f"Invalid jinja parse data: {data}. Must be in the form 'key=value'"
            )
        if key in jinja_parse_data:
            raise ValueError(f"Duplicate key in jinja parse data: {key}")
        jinja_parse_data[key] = value
    return jinja_parse_data


def main():
    args = parse_args()
    input_files = args.input_files
    apps = [args.app] + ([args.a] if args.a else [])

    if args.list_components:
        assert (
            "accelergy" in apps
        ), "Cannot list components without 'accelergy' argument."
        if not input_files:
            os.system("accelergy -l")
            return

    print(f"Running apps: {', '.join(apps)}")
    tl = get_version(input_files)

    if tl is None or any(
        os.path.basename(f) == "parsed-processed-input.yaml" for f in input_files
    ):
        import pytimeloop.timeloopfe.v4 as tl

        call_no_parse(apps, args, tl)
        return

    spec = tl.Specification.from_yaml_files(
        input_files, jinja_parse_data=get_jinja_parse_data(args)
    )
    for app in apps:
        if app == "accelergy" or args.list_components:
            extra_args = ["-l"] if args.list_components else []
            tl.call_accelergy_verbose(
                spec, output_dir=args.output_dir, extra_args=extra_args
            )
        elif app == "model":
            tl.call_model(spec, output_dir=args.output_dir)
        elif app == "mapper":
            tl.call_mapper(spec, output_dir=args.output_dir)
