from util import view_stat

import argparse

parser = argparse.ArgumentParser(description='View benchmark reports.')
parser.add_argument('benchmarks', nargs='+', help='Benchmark names.')
parser.add_argument('--limit', type=float,
                    help='Fraction of functions to print')

if __name__ == '__main__':
    args = parser.parse_args()
    view_stat(args.benchmarks, args.limit)
