"""
Example of using the class `pytimeloop.app.Model`. This is the default way for 
implementing functionality of the `timeloop-model` application.
"""

from pytimeloop.app import Model
from pytimeloop import Config

if __name__ == '__main__':
    import glob

    prefix = '../tests/timeloop-accelergy-exercises/exercises/timeloop/00-model-conv1d-1level/'
    input_files = []
    for input_dir in ['arch/', 'map/', 'prob/']:
        input_files += glob.glob(prefix + input_dir + '*')
    yaml_str = ''
    for fname in input_files:
        with open(fname, 'r') as f:
            yaml_str += f.read()
    config = Config.load_yaml(yaml_str)

    app = Model(config, '.')
    app.run()
