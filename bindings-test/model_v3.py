import sys

import yaml
from bindings import (NativeConfigNode, invoke_accelergy,
                      ArchProperties)
from pytimeloop import (Accelerator, Config, ArchSpecs,
                        Workload, ArchConstraints, Mapping)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class TimeloopModelApp:
    def __init__(self, cfg: Config, out_dir: str, verbose=False,
                 auto_bypass_on_failure=False, out_prefix=''):
        # Use defaults for now
        self.verbose = verbose
        self.auto_bypass_on_failure = auto_bypass_on_failure
        self.out_prefix = out_prefix
        semi_qualified_prefix = 'timeloop-model'
        self.out_prefix = out_dir + '/' + semi_qualified_prefix

        # TODO: print banner if verbose

        # Architecture configuration
        self.arch_specs = ArchSpecs(cfg['architecture'])
        self.arch_specs.generate_tables(
            cfg, semi_qualified_prefix, out_dir, self.out_prefix)

        # Problem configuration
        self.workload = Workload(cfg['problem'])
        if self.verbose:
            print('Problem configuration complete.')

        self.arch_props = ArchProperties(self.arch_specs)

        # Architecture constraints
        self.constraints = ArchConstraints(
            self.arch_props, self.workload, cfg['architecture_constraints'])

        if verbose:
            print('Architecture configuration complete.')

        # Mapping configuration
        self.mapping = Mapping(cfg['mapping'], self.arch_specs, self.workload)
        if verbose:
            print('Mapping construction complete.')

        # Validate mapping against architecture constraints
        if not self.constraints.satisfied_by(self.mapping):
            print('ERROR: mapping violates architecture constraints.')
            exit(1)

    def run(self):
        stats_fname = self.out_prefix + 'stats.txt'
        xml_fname = self.out_prefix + '.map+stats.xml'
        map_txt_fname = self.out_prefix + '.map.txt'

        engine = Accelerator(self.arch_specs)

        eval_stat = engine.evaluate(
            self.mapping, self.workload, False, True, True)


if __name__ == '__main__':
    import glob

    prefix = '../timeloop-accelergy-exercises/exercises/timeloop/00-model-conv1d-1level/'
    input_files = []
    for input_dir in ['arch/', 'map/', 'prob/']:
        input_files += glob.glob(prefix + input_dir + '*')
    yaml_str = ''
    for fname in input_files:
        with open(fname, 'r') as f:
            yaml_str += f.read()
    config = Config.load_yaml(yaml_str)

    app = TimeloopModelApp(config, '.', verbose=True)
    app.run()
