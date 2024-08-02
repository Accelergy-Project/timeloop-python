import glob
from typing import List, Tuple
import unittest
import os


from pytimeloop.timeloopfe.v4.specification import Specification

PROBLEM_FILE = "problem.yaml"
MAPPER_FILE = "mapper_quick.yaml"

ENV_VARS = {
    "TIMELOOP_OUTPUT_STAT_SCIENTIFIC": "1",
    "TIMELOOP_OUTPUT_STAT_DEFAULT_FLOAT": "0",
    "TIMELOOP_ENABLE_FIRST_READ_ELISION": "0",
}
ENVSTR = " ".join(k + "=" + v for k, v in ENV_VARS.items())


class TestLoadExamples(unittest.TestCase):
    def _gather_input_files(self, start_dir) -> Tuple[List[str], List[str]]:
        files = []
        problem = None
        start_dir = os.path.join("arch_spec_examples", start_dir)
        mapper = os.path.join("arch_spec_examples", "mapper_quick.yaml")
        variables = os.path.join("arch_spec_examples", "variables.yaml")
        for f in os.listdir(start_dir):
            if "arch" in f:
                continue
            elif "problem" in f:
                problem = os.path.join(start_dir, f)
            else:
                files.append(os.path.join(start_dir, f))
        if problem is None:
            newcheck = f"problem_{os.path.basename(start_dir)}.yaml"
            problem = os.path.join(start_dir, "..", newcheck)
            problem = problem if os.path.exists(problem) else None
        if problem is None:
            newcheck = "problem.yaml"
            problem = os.path.join(start_dir, "..", newcheck)
        if os.path.exists(os.path.join(start_dir, "..", "components")):
            files += glob.glob(os.path.join(start_dir, "..", "components/*.yaml"))
        files += [mapper, problem, variables]
        return (
            [os.path.join(start_dir, "arch_old.yaml")] + files,
            [os.path.join(start_dir, "arch.yaml")] + files,
        )

    def run_test(self, start_dir: str, new_suffix: str) -> None:
        _, f2 = self._gather_input_files(start_dir)
        if new_suffix:
            f2[0] = f2[0].replace(".yaml", new_suffix + ".yaml")

        spec = Specification.from_yaml_files(*f2)
        spec.process()

    def test_eyriss_like(self):
        self.run_test("eyeriss_like", "")

    def test_eyeriss_like_split(self):
        self.run_test("eyeriss_like", "_split")

    def test_simba_like(self):
        self.run_test("simba_like", "")

    def test_simba_like_split(self):
        self.run_test("simba_like", "_split")

    def test_simple_output_stationary(self):
        self.run_test("simple_output_stationary", "")

    def test_simple_output_stationary_split(self):
        self.run_test("simple_output_stationary", "_split")

    def test_simple_pim(self):
        self.run_test("simple_pim", "")

    def test_simple_pim_split(self):
        self.run_test("simple_pim", "_split")

    def test_simple_weight_stationary(self):
        self.run_test("simple_weight_stationary", "")

    def test_simple_weight_stationary_split(self):
        self.run_test("simple_weight_stationary", "_split")

    def test_sparse_tensor_core_like(self):
        self.run_test("sparse_tensor_core_like", "")

    def test_sparse_tensor_core_like_split(self):
        self.run_test("sparse_tensor_core_like", "_split")

    def test_sparseloop_01_2_1_DUDU_dot_product(self):
        self.run_test("sparseloop/01.2.1-DUDU-dot-product", "")

    def test_sparseloop_01_2_1_SUDU_dot_product_split(self):
        self.run_test("sparseloop/01.2.2-SUDU-dot-product", "_split")

    def test_sparseloop_01_2_2_SUDU_dot_product(self):
        self.run_test("sparseloop/01.2.2-SUDU-dot-product", "")

    def test_sparseloop_01_2_2_SUDU_dot_product_split(self):
        self.run_test("sparseloop/01.2.2-SUDU-dot-product", "_split")

    def test_sparseloop_01_2_3_SCDU_dot_product(self):
        self.run_test("sparseloop/01.2.3-SCDU-dot-product", "")

    def test_sparseloop_01_2_3_SCDU_dot_product_split(self):
        self.run_test("sparseloop/01.2.3-SCDU-dot-product", "_split")

    def test_sparseloop_02_2_1_spMspM(self):
        self.run_test("sparseloop/02.2.1-spMspM", "")

    def test_sparseloop_02_2_1_spMspM_split(self):
        self.run_test("sparseloop/02.2.1-spMspM", "_split")

    def test_sparseloop_02_2_2_spMspM_tiled(self):
        self.run_test("sparseloop/02.2.2-spMspM-tiled", "")

    def test_sparseloop_02_2_2_spMspM_tiled_split(self):
        self.run_test("sparseloop/02.2.2-spMspM-tiled", "_split")

    def test_sparseloop_03_2_1_conv1d(self):
        self.run_test("sparseloop/03.2.1-conv1d", "")

    def test_sparseloop_03_2_1_conv1d_split(self):
        self.run_test("sparseloop/03.2.1-conv1d", "_split")

    def test_sparseloop_03_2_2_conv1d_oc(self):
        self.run_test("sparseloop/03.2.2-conv1d+oc", "")

    def test_sparseloop_03_2_2_conv1d_oc_split(self):
        self.run_test("sparseloop/03.2.2-conv1d+oc", "_split")

    def test_sparseloop_03_2_3_conv1d_oc_spatial(self):
        self.run_test("sparseloop/03.2.3-conv1d+oc-spatial", "")

    def test_sparseloop_03_2_3_conv1d_oc_spatial_split(self):
        self.run_test("sparseloop/03.2.3-conv1d+oc-spatial", "_split")

    def test_sparseloop_04_2_1_eyeriss_like_gating(self):
        self.run_test("sparseloop/04.2.1-eyeriss-like-gating", "")

    def test_sparseloop_04_2_1_eyeriss_like_gating_split(self):
        self.run_test("sparseloop/04.2.1-eyeriss-like-gating", "_split")

    def test_sparseloop_04_2_2_eyeriss_like_gating_mapspace_search(self):
        self.run_test(
            "sparseloop/04.2.2-eyeriss-like-gating-mapspace-search",
            "",
        )

    def test_sparseloop_04_2_2_eyeriss_like_gating_mapspace_search_split(
        self,
    ):
        self.run_test(
            "sparseloop/04.2.2-eyeriss-like-gating-mapspace-search",
            "_split",
        )

    def test_sparseloop_04_2_3_eyeriss_like_onchip_compression(self):
        self.run_test("sparseloop/04.2.3-eyeriss-like-onchip-compression", "")

    def test_sparseloop_04_2_3_eyeriss_like_onchip_compression_split(self):
        self.run_test(
            "sparseloop/04.2.3-eyeriss-like-onchip-compression",
            "_split",
        )
