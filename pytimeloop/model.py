import bindings
from .accelergy_interface import invoke_accelergy
from .config import Config

import logging


class ArchSpecs(bindings.model.ArchSpecs):
    def __init__(self, config, is_sparse_topology: bool = False):
        super().__init__(config, is_sparse_topology)

    def generate_tables(self, config: Config, semi_qualified_prefix, out_dir,
                        out_prefix, log_level=logging.INFO):
        # Setup logger
        logger = logging.getLogger(__name__ + '.' + __class__.__name__)
        logger.setLevel(log_level)

        root_node = config.get_root()
        if 'ERT' in root_node:
            logger.info('Found Accelergy ERT, replacing internal energy model')
            self.parse_accelergy_ert(root_node['ERT'])
            if 'ART' in root_node:
                logger.info(
                    'Found Accelergy ART, replacing internal area model')
                self.parse_accelergy_art(root_node['ART'])
        else:
            arch_cfg = root_node['architecture']
            if 'subtree' in arch_cfg or 'local' in arch_cfg:
                with open('tmp-accelergy.yaml', 'w+') as f:
                    f.write(config.dump_yaml_str())
                result = invoke_accelergy(['tmp-accelergy.yaml'],
                                          semi_qualified_prefix, out_dir)
                logger.info('Generated Accelergy ERT to replace internal '
                            'energy model')
                self.parse_accelergy_ert(result.ert)

                logger.info('Generated Accelergy ART to replace internal '
                            'energy model')
                self.parse_accelergy_art(result.art)


class SparseOptimizationInfo(bindings.model.SparseOptimizationInfo):
    def __init__(self, sparse_config, arch_specs: ArchSpecs):
        super().__init__(sparse_config, arch_specs)

