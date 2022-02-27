import bindings
from .accelergy_interface import invoke_accelergy
from .config import Config

import logging


class ArchSpecs(bindings.ArchSpecs):
    def __init__(self, config: Config, is_sparse_topology: bool=False):
        _, native_arch_node = config.get_native()
        super().__init__(native_arch_node, is_sparse_topology)

    def generate_tables(self, config: Config, semi_qualified_prefix, out_dir,
                        out_prefix, log_level=logging.INFO):
        # Setup logger
        logger = logging.getLogger(__name__ + '.' + __class__.__name__)
        logger.setLevel(log_level)

        native_root_cfg, native_cfg = config.get_native()
        root_node = native_root_cfg.get_root()
        if 'ERT' in root_node:
            logger.info('Found Accelergy ERT, replacing internal energy model')
            self.parse_accelergy_ert(root_node['ert'])
            if 'ART' in root_node:
                logger.info(
                    'Found Accelergy ART, replacing internal area model')
                self.parse_accelergy_art(root_node['art'])
        else:
            _, native_arch_cfg = config['architecture'].get_native()
            if 'subtree' in native_arch_cfg or 'local' in native_arch_cfg:
                with open('tmp-accelergy.yaml', 'w+') as f:
                    f.write(config.dump_yaml())
                invoke_accelergy(['tmp-accelergy.yaml'],
                                 semi_qualified_prefix, out_dir)
                ert_path = out_prefix + '.ERT.yaml'
                # Have to store config in a variable, so it doesn't get
                # garbage collected. CompoundConfigNode referes to it.
                ert_cfg = bindings.Config(ert_path)
                ert = ert_cfg.get_root().lookup('ERT')
                logger.info('Generated Accelergy ERT to replace internal '
                            'energy model')
                self.parse_accelergy_ert(ert)

                art_path = out_prefix + '.ART.yaml'
                art_cfg = bindings.Config(art_path)
                art = art_cfg.get_root()['ART']
                logger.info('Generated Accelergy ART to replace internal '
                            'energy model')
                self.parse_accelergy_art(art)


class SparseOptimizationInfo(bindings.SparseOptimizationInfo):
    def __init__(self, sparse_config: Config, arch_specs: ArchSpecs):
        _, native_sparse_config_node = sparse_config.get_native()
        super().__init__(native_sparse_config_node, arch_specs)
