#!/usr/bin/env python3
from yaml import parse
from pytimeloop.app import Model
from pytimeloop import Config

import argparse
import glob
import logging

parser = argparse.ArgumentParser(
    description='Run Timeloop given architecture, workload, and mapping.')
parser.add_argument('configs', nargs='+', help='Config files to run Timeloop.')
parser.add_argument('--verbosity', type=int, default=1,
                    help='0 is only error; 1 adds warning; 2 is everyting.')


def load_configs(input_fnames):
    input_files = []
    for fname in input_fnames:
        input_files += glob.glob(fname)
    yaml_str = ''
    for fname in input_files:
        with open(fname, 'r') as f:
            yaml_str += f.read()
    config = Config.load_yaml(yaml_str)
    return config


if __name__ == '__main__':
    args = parser.parse_args()
    config = load_configs(args.configs)

    log_level = logging.INFO
    if args.verbosity == 0:
        log_level = logging.INFO
    elif args.verbosity == 1:
        log_level = logging.WARNING
    elif args.verbosity == 2:
        log_level = logging.INFO
    else:
        raise ValueError('Verbosity level unrecognized.')

    app = Model(config, '.')
    app.run()
