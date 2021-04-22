from bindings import (NativeArchSpecs, NativeConfig,
                      NativeConfigNode, invoke_accelergy)
from .config import Config


class ArchSpecs(NativeArchSpecs):
    def __init__(self, config: Config, semi_qualified_prefix, out_dir,
                 out_prefix, verbose=True):
        _, native_arch_node = config.get_native()
        super().__init__(native_arch_node)
        self.verbose = verbose
        self.generate_tables(
            config, semi_qualified_prefix, out_dir, out_prefix)

    def generate_tables(self, config: Config, semi_qualified_prefix, out_dir,
                        out_prefix):
        native_cfg, native_arch_cfg = config.get_native()
        root_node = native_cfg.get_root()
        if 'ERT' in root_node:
            if self.verbose:
                print('Found Accelergy ERT, replacing internal energy model')
            self.parse_accelergy_ert(root_node['ert'])
            if 'ART' in root_node:
                if self.verbose:
                    print('Found Accelergy ART, replacing internal area model')
                self.parse_accelergy_art(root_ndoe['art'])
        else:
            if 'subtree' in native_arch_cfg or 'local' in native_arch_cfg:
                print('Invoking Accelergy')
                with open('tmp-accelergy.yaml', 'w+') as f:
                    f.write(config.dump_yaml())
                invoke_accelergy(['tmp-accelergy.yaml'],
                                 semi_qualified_prefix, out_dir)
                ert_path = out_prefix + '.ERT.yaml'
                # Have to store config in a variable, so it doesn't get
                # garbage collected. CompoundConfigNode referes to it.
                ert_cfg = NativeConfig(ert_path)
                ert = ert_cfg.get_root().lookup('ERT')
                if self.verbose:
                    print('Generated Accelergy ERT to replace internal energy '
                          'model')
                self.parse_accelergy_ert(ert)

                art_path = out_prefix + '.ART.yaml'
                art_cfg = NativeConfig(art_path)
                art = art_cfg.get_root()['ART']
                if self.verbose:
                    print('Generated Accelergy ART to replace internal energy '
                          'model')
                self.parse_accelergy_art(art)
