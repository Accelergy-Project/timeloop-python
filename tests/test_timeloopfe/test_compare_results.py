import glob
import shutil
import threading
from typing import List, Tuple
import unittest
import os
import logging
from accelergy.utils import yaml

from pytimeloop.timeloopfe.v4.specification import Specification as spec4
from pytimeloop.timeloopfe.v3.specification import Specification as spec3
from pytimeloop.timeloopfe.common.backend_calls import call_mapper, call_model, to_mapper_app, to_model_app

from pytimeloop.timeloopfe.v4.processors import (
    References2CopiesProcessor,
    ConstraintAttacherProcessor,
    ConstraintMacroProcessor,
    Dataspace2BranchProcessor,
    EnableDummyTableProcessor,
    SparseOptAttacherProcessor,
)

PROBLEM_FILE = "problem.yaml"
MAPPER_FILE = "mapper_quick.yaml"

ENV_VARS = {
    "TIMELOOP_OUTPUT_STAT_SCIENTIFIC": "1",
    "TIMELOOP_OUTPUT_STAT_DEFAULT_FLOAT": "0",
    "TIMELOOP_ENABLE_FIRST_READ_ELISION": "0",
}
ENVSTR = " ".join(k + "=" + v for k, v in ENV_VARS.items())
PROCESSORS = [
    References2CopiesProcessor,
    ConstraintAttacherProcessor,
    SparseOptAttacherProcessor,
    ConstraintMacroProcessor,
    Dataspace2BranchProcessor,
    EnableDummyTableProcessor,
]


class TestCompareResults(unittest.TestCase):
    def _grabstats_from_file(self, path: str) -> str:
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist. Check if Timeloop ran.")
        contents = open(os.path.realpath(path)).read()
        # Grab contents between lines containing "Summary Stats" and "Computes"
        return self._grabstats(contents)
    
    def _grabstats(self, contents: str) -> str:
        start, end = None, None
        for i, line in enumerate(contents.split("\n")):
            if "Summary Stats" in line:
                start = i
            if start is not None and "Computes" in line:
                end = i
                break
        return "\n".join(contents.split("\n")[start:end])

    def compare_logs(self, log1: str, log2: str) -> None:
        o1 = open(log1).readlines()
        o2 = open(log2).readlines()
        headings = [
            "Factorization options",
            "LoopPermutation",
            "Spatial",
            "DatatypeBypass",
        ]
        o1 = [x for x in enumerate(o1) if any(h in x[1] for h in headings)]
        o2 = [x for x in enumerate(o2) if any(h in x[1] for h in headings)]
        for (i1, l1), (i2, l2) in zip(o1, o2):
            diffstr = f"{log1} line {i1} differs from {log2} line {i2}"
            self.assertEqual(l1, l2, diffstr)

    def run_timeloop(self, targets: List[str], run_model: bool) -> None:
        tl_cmd = "timeloop-model" if run_model else "timeloop-mapper"
        log_output = ">> output.log 2>&1"

        def cmd(d):
            if os.path.isdir(d):
                t = os.path.join(d, "inputs", "*.yaml")
            else:
                t = os.path.abspath(d)
                d = os.path.dirname(os.path.dirname(d))
            logging.info("Running: cd %s ; %s %s %s", d, ENVSTR, tl_cmd, t)
            os.system(f"cd {d} ; {ENVSTR} {tl_cmd} {t} {log_output}")

        threads = [threading.Thread(target=cmd, args=(t,)) for t in targets]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

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

    def run_test(self, start_dir: str, new_suffix: str, run_model: bool) -> None:
        f1, f2 = self._gather_input_files(start_dir)
        if new_suffix:
            f2[0] = f2[0].replace(".yaml", new_suffix + ".yaml")

        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        start_dir += new_suffix
        for f, target in [(f1, "old"), (f2, "new")]:
            d = os.path.join(this_script_dir, "compare", start_dir, target)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(os.path.join(d, "inputs"), exist_ok=True)
            for f_ in f:
                shutil.copy(f_, os.path.join(d, "inputs"))

        # Run the preprocdataspacesor
        olddir = os.path.join(this_script_dir, "compare", start_dir, "old")
        newdir = os.path.join(this_script_dir, "compare", start_dir, "new")
        s0 = spec3.from_yaml_files(f"{olddir}/inputs/*.yaml")
        s1 = spec4.from_yaml_files(f"{newdir}/inputs/*.yaml")
        del s1.mapspace.template
        while s1._required_processors:
            s1._required_processors.pop()
        s1._required_processors.extend(PROCESSORS)
        for s, d in [(s0, olddir), (s1, newdir)]:
            s.process()
            dumpto = f"{d}/inputs/arch.yaml"
            logto = f"{d}/output.log"
            f = call_model if run_model else call_mapper
            f(s, d, dump_intermediate_to=dumpto, log_to=logto)

            if run_model:
                pytimeloop_stats = to_model_app(s, d).run().stats_string
            else:
                pytimeloop_stats = to_mapper_app(s, d).run().stats_string

        # spec = Specification.from_yaml_files(
        #     f"{newdir}/inputs/*.yaml", processors=PROCESSORS
        # )
        # spec.process()
        # newarch = os.path.join(newdir, "inputs/arch.yaml")
        # yaml.write_yaml_file(newarch, transpile(spec, run_model))
        # self.run_timeloop([olddir, newarch], run_model)

        # Compare the results
        self.compare_logs(
            os.path.join(olddir, "output.log"),
            os.path.join(newdir, "output.log"),
        )
        x = "mapper" if not run_model else "model"
        stats1 = os.path.join(newdir, f"timeloop-{x}.stats.txt")
        stats2 = os.path.join(olddir, f"timeloop-{x}.stats.txt")
        self.assertEqual(self._grabstats_from_file(stats1),
                         self._grabstats_from_file(stats2))
        self.assertEqual(self._grabstats_from_file(stats1),
                         self._grabstats(pytimeloop_stats))

    def test_eyriss_like(self):
        self.run_test("eyeriss_like", "", False)

    def test_eyeriss_like_split(self):
        self.run_test("eyeriss_like", "_split", False)

    def test_simba_like(self):
        self.run_test("simba_like", "", False)

    def test_simba_like_split(self):
        self.run_test("simba_like", "_split", False)

    def test_simple_output_stationary(self):
        self.run_test("simple_output_stationary", "", False)

    def test_simple_output_stationary_split(self):
        self.run_test("simple_output_stationary", "_split", False)

    def test_simple_pim(self):
        self.run_test("simple_pim", "", False)

    def test_simple_pim_split(self):
        self.run_test("simple_pim", "_split", False)

    def test_simple_weight_stationary(self):
        self.run_test("simple_weight_stationary", "", False)

    def test_simple_weight_stationary_split(self):
        self.run_test("simple_weight_stationary", "_split", False)

    def test_sparse_tensor_core_like(self):
        self.run_test("sparse_tensor_core_like", "", False)

    def test_sparse_tensor_core_like_split(self):
        self.run_test("sparse_tensor_core_like", "_split", False)

    def test_sparseloop_01_2_1_DUDU_dot_product(self):
        self.run_test("sparseloop/01.2.1-DUDU-dot-product", "", True)

    def test_sparseloop_01_2_1_SUDU_dot_product_split(self):
        self.run_test("sparseloop/01.2.2-SUDU-dot-product", "_split", True)

    def test_sparseloop_01_2_2_SUDU_dot_product(self):
        self.run_test("sparseloop/01.2.2-SUDU-dot-product", "", True)

    def test_sparseloop_01_2_2_SUDU_dot_product_split(self):
        self.run_test("sparseloop/01.2.2-SUDU-dot-product", "_split", True)

    def test_sparseloop_01_2_3_SCDU_dot_product(self):
        self.run_test("sparseloop/01.2.3-SCDU-dot-product", "", True)

    def test_sparseloop_01_2_3_SCDU_dot_product_split(self):
        self.run_test("sparseloop/01.2.3-SCDU-dot-product", "_split", True)

    def test_sparseloop_02_2_1_spMspM(self):
        self.run_test("sparseloop/02.2.1-spMspM", "", True)

    def test_sparseloop_02_2_1_spMspM_split(self):
        self.run_test("sparseloop/02.2.1-spMspM", "_split", True)

    def test_sparseloop_02_2_2_spMspM_tiled(self):
        self.run_test("sparseloop/02.2.2-spMspM-tiled", "", True)

    def test_sparseloop_02_2_2_spMspM_tiled_split(self):
        self.run_test("sparseloop/02.2.2-spMspM-tiled", "_split", True)

    def test_sparseloop_03_2_1_conv1d(self):
        self.run_test("sparseloop/03.2.1-conv1d", "", True)

    def test_sparseloop_03_2_1_conv1d_split(self):
        self.run_test("sparseloop/03.2.1-conv1d", "_split", True)

    def test_sparseloop_03_2_2_conv1d_oc(self):
        self.run_test("sparseloop/03.2.2-conv1d+oc", "", True)

    def test_sparseloop_03_2_2_conv1d_oc_split(self):
        self.run_test("sparseloop/03.2.2-conv1d+oc", "_split", True)

    def test_sparseloop_03_2_3_conv1d_oc_spatial(self):
        self.run_test("sparseloop/03.2.3-conv1d+oc-spatial", "", True)

    def test_sparseloop_03_2_3_conv1d_oc_spatial_split(self):
        self.run_test("sparseloop/03.2.3-conv1d+oc-spatial", "_split", True)

    def test_sparseloop_04_2_1_eyeriss_like_gating(self):
        self.run_test("sparseloop/04.2.1-eyeriss-like-gating", "", True)

    def test_sparseloop_04_2_1_eyeriss_like_gating_split(self):
        self.run_test("sparseloop/04.2.1-eyeriss-like-gating", "_split", True)

    # def test_sparseloop_04_2_2_eyeriss_like_gating_mapspace_search(self):
    #     self.run_test(
    #         "sparseloop/04.2.2-eyeriss-like-gating-mapspace-search",
    #         "",
    #         False,
    #     )

    # def test_sparseloop_04_2_2_eyeriss_like_gating_mapspace_search_split(
    #     self,
    # ):
    #     self.run_test(
    #         "sparseloop/04.2.2-eyeriss-like-gating-mapspace-search",
    #         "_split",
    #         False,
    #     )

    def test_sparseloop_04_2_3_eyeriss_like_onchip_compression(self):
        self.run_test("sparseloop/04.2.3-eyeriss-like-onchip-compression", "", False)

    def test_sparseloop_04_2_3_eyeriss_like_onchip_compression_split(self):
        self.run_test(
            "sparseloop/04.2.3-eyeriss-like-onchip-compression",
            "_split",
            False,
        )
