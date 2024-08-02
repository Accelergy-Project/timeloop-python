import os
import sys
import logging
import pytimeloop.timeloopfe.v4 as tl
from pytimeloop.timeloopfe.v4.processors import EnableDummyTableProcessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    OPTIONS = {
        0: "eyeriss_like",
        1: "eyeriss_nested_hierarchy",
        2: "simba_like",
        3: "simple_output_stationary",
        4: "simple_pim",
        5: "simple_weight_stationary",
        6: "sparse_tensor_core_like",
        7: "sparseloop/01.2.1-DUDU-dot-product",
        8: "sparseloop/01.2.2-SUDU-dot-product",
        9: "sparseloop/01.2.3-SCDU-dot-product",
        10: "sparseloop/01.2.1-DUDU-dot-product",
        11: "sparseloop/02.2.1-spMspM",
        12: "sparseloop/02.2.2-spMspM-tiled",
        13: "sparseloop/03.2.1-conv1d",
        14: "sparseloop/03.2.2-conv1d+oc",
        15: "sparseloop/03.2.3-conv1d+oc-spatial",
        16: "sparseloop/04.2.1-eyeriss-like-gating",
        17: "sparseloop/04.2.2-eyeriss-like-gating-mapspace-search",
        18: "sparseloop/04.2.3-eyeriss-like-onchip-compression",
        19: "raella",
    }

    CHOICE = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    SPLIT = False

    assert (
        CHOICE in OPTIONS
    ), f'Invalid choice "{CHOICE}". Choose from {list(OPTIONS.keys())}'

    cinstr = "_split" if SPLIT else ""
    TARGET = [f"{OPTIONS[CHOICE]}/arch{cinstr}", "problem", "mapper", "variables"]
    TARGET = [os.path.join("arch_spec_examples", f"{t}.yaml") for t in TARGET]

    alt_prob = f"problem_{OPTIONS[CHOICE]}"
    if os.path.exists(TARGET[1].replace("problem", alt_prob)):
        TARGET[1] = TARGET[1].replace("problem", alt_prob)

    # Add in some extra files for Sparseloop inputs
    if "sparseloop" in OPTIONS[CHOICE]:
        TARGET = TARGET[:1]
        for f in os.listdir(os.path.dirname(TARGET[0])):
            if "arch" not in f:
                TARGET.append(os.path.join(os.path.dirname(TARGET[0]), f))

    for_model = CHOICE >= 7 and CHOICE != 17 and CHOICE != 19

    spec = tl.Specification.from_yaml_files(*TARGET)
    spec.process()
    # Add in the EnableDummyTable processor so Accelergy provides dummy numbers
    # and we don't have to worry about having area/energy estimators
    spec.process(EnableDummyTableProcessor)

    # Run the mapper or model
    if for_model:
        tl.call_model(spec, "./outdir")
    else:
        tl.call_mapper(spec, "./outdir")
