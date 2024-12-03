import logging
from pathlib import Path

from bindings.config import Config

from pytimeloop.fastfusion.mapper.mapper_snowcat import mapper
from pytimeloop.fastfusion.mapper.simexplore import explore_fusion
from pytimeloop.fastfusion.plot.ski_slope import plot_ski_slope
from pytimeloop.timeloopfe.v4fused import Specification

from tests.util import CONFIG_DIR, TEST_TMP_DIR


class MhaExperiment:
    WORKLOAD_TEMPLATE = "cascaded_mha.workload_template.yaml"
    def __init__(self):
        workload_fname = CONFIG_DIR / MhaExperiment.WORKLOAD_TEMPLATE
        with open(workload_fname, "r", encoding="utf-8") as f:
            self.workload_template = f.read()

    def configure_workload_shape(self, **kwargs):
        self.workload_config = self.workload_template.format(**kwargs)

    def configure_arch(self, arch_fname: str | Path):
        arch_fname = CONFIG_DIR / arch_fname
        with open(arch_fname, "r") as f:
            self.arch_config = f.read()

    def run_experiment(self):
        config_str = self.workload_config + "\n" + self.arch_config
        config = Config(config_str, "yaml")
        with open(TEST_TMP_DIR / "tmp.yaml", "w") as f:
            f.write(config_str)
        spec = Specification.from_yaml_files([TEST_TMP_DIR / "tmp.yaml"])
        result = mapper(config,
                        explore_glb_uneven=True,
                        spec=spec,
                        tmp_path=TEST_TMP_DIR)
        self.data = explore_fusion(result)

    def plot_ski_slope(self, **kwargs):
        fig, ax = plot_ski_slope(self.data, **kwargs)
        fig.tight_layout()
        return fig, ax


def main():
    logging.basicConfig(filename='fastfusion.mha.log', level=logging.DEBUG)
    exp = MhaExperiment()
    exp.configure_workload_shape(B=1, H=1, M=1, E=1, D=1)
    exp.configure_arch("snowcat.arch.yaml")
    exp.run_experiment()
    fig, ax = exp.plot_ski_slope()
    fig.savefig("mha_shape0.skislope.png", dpi=200)


if __name__ == '__main__':
    main()