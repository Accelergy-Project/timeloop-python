from yaml import parse
from pytimeloop.app import MapperApp
from pytimeloop import Config

import glob
import os
from pathlib import Path
import pstats

PROJECT_DIR = Path(__file__).parent.parent
TIMELOOP_EXAMPLES_DIR = (
    PROJECT_DIR / 'tests/timeloop-accelergy-exercises/workspace/'
    / 'exercises/2020.ispass/timeloop')
BENCH_DIR = Path(__file__).parent / 'benchfiles'


def load_configs(input_fnames):
    input_files = []
    for fname in input_fnames:
        input_files += glob.glob(fname)
    yaml_str = ''
    for fname in input_files:
        with open(fname, 'r') as f:
            yaml_str += f.read()
        yaml_str += '\n'
    config = Config.load_yaml(yaml_str)
    return config


def pytimeloop_bench(bench_name, bench_fname=None):
    if not os.path.isdir(BENCH_DIR):
        os.mkdir(BENCH_DIR)

    bench_dir = BENCH_DIR / bench_name
    if not os.path.isdir(bench_dir):
        os.mkdir(bench_dir)

    if bench_fname is None:
        bench_fname = bench_name + '.pstat'

    def dec(bench_f):
        return lambda: bench_f(bench_dir, bench_dir / bench_fname)

    return dec


def view_stat(bench_names, *restrictions):
    for name in bench_names:
        stat_path = BENCH_DIR / name / (name + '.pstat')
        stat = pstats.Stats(str(stat_path))
        stat.sort_stats('cumtime')
        stat.print_stats(*restrictions)
