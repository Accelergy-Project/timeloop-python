from collections import defaultdict
import logging.handlers
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from ruamel.yaml import YAML
from joblib import Parallel, delayed

yaml = YAML(typ="safe")

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.looptree.equivalent_ranks import EquivalentGroups

from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.layerdeduplication import is_equivalent
from pytimeloop.fastfusion.mapper.logging import make_queue_and_listener
from pytimeloop.fastfusion.mapper.per_einsum_mapper import get_top_loop_jobs, mapper_place_fusion_level

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

    einsum_name_to_id = workload.einsum_name_to_id()

    if isinstance(tmp_path, Path):
        tmp_path = str(tmp_path)
    call_accelergy_verbose(spec, tmp_path)
    ert_dict = yaml.load(Path(tmp_path) / "ERT.yaml")
    ert = Ert(ert_dict["ERT"])
    energy_dict = ert.to_dict()

    args = get_top_loop_jobs(
        einsum_name_to_id=einsum_name_to_id,
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
    
    print(f'Number of jobs: {len(args)}')
    
    logger.debug("Starting workers")
    log_queue_listener.start()
    
    result = Parallel(n_jobs=128)(
        delayed(mapper_place_fusion_level)(**a) for a in args
    )
    data = defaultdict(dict)
    for einsum_id, mappings in result:
        for k, v in mappings.items():
            if k in data[einsum_id]:
                data[einsum_id][k] += v
            else:
                data[einsum_id][k] = v
        
    log_queue_listener.stop()
    logger.info(f"Mapper finished for {spec}")
    return data

