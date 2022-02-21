#!/usr/bin/env python3
from yaml import parse
from pytimeloop.app import MapperApp
from pytimeloop import Config

import cProfile
import pstats

from util import (BENCH_DIR, TIMELOOP_EXAMPLES_DIR,
                  load_configs, pytimeloop_bench)

TEST_DIR = TIMELOOP_EXAMPLES_DIR / '05-mapper-conv1d+oc-3level'


def load_mapper_files():
    arch_fnames = str(TEST_DIR / 'arch/*')
    constraint_fname = str(
        TEST_DIR / 'constraints/null.constraints.yaml')
    mapper_fname = str(TEST_DIR / 'mapper/*')
    prob_fname = str(TEST_DIR / 'prob/*')

    config = load_configs(
        [arch_fnames, constraint_fname, mapper_fname, prob_fname])

    return config


@pytimeloop_bench('mapper_1thread')
def mapper_bench_1thread(bench_dir, bench_fname):
    config = load_mapper_files()

    app = MapperApp(config, str(bench_dir))
    eval_stats = None
    cProfile.runctx('_ = app.run()', {
                    'app': app, 'eval_stats': eval_stats}, {}, bench_fname)

    p = pstats.Stats(str(bench_fname))
    p.print_stats()


@pytimeloop_bench('mapper_2thread')
def mapper_bench_2thread(bench_dir, bench_fname):
    config = load_mapper_files()
    config['mapper']['num-threads'] = 2

    app = MapperApp(config, str(bench_dir))
    eval_stats = None
    cProfile.runctx('eval_stats, _ = app.run()', {
                    'app': app, 'eval_stats': eval_stats}, {}, bench_fname)

    p = pstats.Stats(str(bench_fname))
    p.print_stats()


@pytimeloop_bench('mapper_4thread')
def mapper_bench_4thread(bench_dir, bench_fname):
    config = load_mapper_files()
    config['mapper']['num-threads'] = 4

    app = MapperApp(config, str(bench_dir))
    eval_stats = None
    cProfile.runctx('eval_stats, _ = app.run()', {
                    'app': app, 'eval_stats': eval_stats}, {}, bench_fname)

    p = pstats.Stats(str(bench_fname))
    p.print_stats()


@pytimeloop_bench('mapper_8thread')
def mapper_bench_8thread(bench_dir, bench_fname):
    config = load_mapper_files()
    config['mapper']['num-threads'] = 8

    app = MapperApp(config, str(bench_dir))
    eval_stats = None
    cProfile.runctx('eval_stats, _ = app.run()', {
                    'app': app, 'eval_stats': eval_stats}, {}, bench_fname)

    p = pstats.Stats(str(bench_fname))
    p.print_stats()


if __name__ == '__main__':
    mapper_bench_1thread()
    mapper_bench_2thread()
    mapper_bench_4thread()
    mapper_bench_8thread()
