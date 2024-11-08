import logging.handlers
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from ruamel.yaml import YAML

yaml = YAML(typ="safe")

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.looptree.equivalent_ranks import EquivalentGroups

from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.mapper.logging import make_queue_and_listener
from pytimeloop.fastfusion.mapper.per_einsum_mapper import mapper_one_einsum

from pytimeloop.timeloopfe.v4 import Ert
from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose


def mapper(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    explore_glb_uneven,
    explore_pe_uneven,
    spec,
    tmp_path,
    verbose_stream=None,
):
    logger.info(f"Calling mapper for {spec}")

    log_queue, log_queue_listener = make_queue_and_listener()

    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    einsum_name_to_id = workload.einsum_name_to_id()

    if isinstance(tmp_path, Path):
        tmp_path = str(tmp_path)
    call_accelergy_verbose(spec, tmp_path)
    ert_dict = yaml.load(Path(tmp_path) / "ERT.yaml")
    ert = Ert(ert_dict["ERT"])
    energy_dict = ert.to_dict()

    data = {}
    per_einsum_args = [
        dict(
            einsum_id=einsum_id,
            config=config,
            pe_array_constraint=pe_array_constraint,
            mac_array_constraint=mac_array_constraint,
            explore_glb_uneven=explore_glb_uneven,
            explore_pe_uneven=explore_pe_uneven,
            spec=spec,
            energy_dict=energy_dict,
            log_queue=log_queue,
            verbose_stream=verbose_stream,
        )
        for einsum_id in einsum_name_to_id.values()
    ]

    from joblib import Parallel, delayed

    logger.debug("Starting workers")
    log_queue_listener.start()
    data = Parallel(n_jobs=32)(
        delayed(mapper_one_einsum)(**args) for args in per_einsum_args
    )
    log_queue_listener.stop()

    data = {
        einsum_id: mapping
        for einsum_id, mapping in zip(einsum_name_to_id.values(), data)
    }
    logger.info(f"Mapper finished for {spec}")
    return data

