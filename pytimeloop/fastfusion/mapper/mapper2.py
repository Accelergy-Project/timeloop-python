from collections import defaultdict
from copy import deepcopy
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
from pytimeloop.fastfusion.sim import Tiling, Loop, TensorStorage
from pytimeloop.fastfusion.pareto import MAPPING

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

    grouped_similar_einsums = convert_rank_to_group_renaming(
        detect_similar_einsums(workload, analyzer),
        equivalent_groups
    )
    logger.info(f"Found {len(grouped_similar_einsums)} unique Einsums\n"
                + f"\tConverter: {grouped_similar_einsums}")

    args = get_top_loop_jobs(
        einsums_to_explore=list(grouped_similar_einsums.keys()),
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
    n_workers = 128
    logger.debug(f"Starting {n_workers} workers")
    log_queue_listener.start()
    
    result = Parallel(n_jobs=n_workers)(
        delayed(mapper_place_fusion_level)(**a) for a in args
    )
    data = defaultdict(dict)
    total = 0
    for einsum_id, mappings, count in result:
        for k, v in mappings.items():
            if k in data[einsum_id]:
                data[einsum_id][k] += v
            else:
                data[einsum_id][k] = v
                
        total += count
    print(f"Total number of mappings: {total}")
        
    log_queue_listener.stop()
    logger.info(f"Mapper finished for {spec}")

    generated_data = {}
    logger.info(f"Generating data for non-unique Einsums")
    for from_einsum, others in grouped_similar_einsums.items():
        for to_einsum, (rank_renaming, tensor_renaming) in others.items():
            logger.info(f"Generating data for {to_einsum}. "
                        + f"Rank renaming={rank_renaming}. "
                        + f"Tensor renaming={tensor_renaming}")
            generated_data[to_einsum] = generate_data(from_einsum,
                                                         to_einsum,
                                                         data[from_einsum],
                                                         rank_renaming,
                                                         tensor_renaming)
            

    for einsum, mapping in generated_data.items():
        data[einsum] = mapping

    logger.info(f"Final set of Einsums: {set(data.keys())}")

    # data has to come out in sorted Einsum-id order
    data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}

    return data


def generate_data(from_einsum: int, to_einsum: int, data, rank_renaming, tensor_renaming):
    return {
        _convert_tiling(tiling, rank_renaming, tensor_renaming)
        :
        _convert_stats(from_einsum, to_einsum, stats, rank_renaming, tensor_renaming)
        for tiling, stats in data.items()
    }


def _convert_tiling(tiling: Tiling, rank_renaming, tensor_renaming):
    return Tiling(
        loops=tuple(Loop(rank_renaming[l.rank_id], l.bound, l.is_spatial)
                    for l in tiling.loops),
        tensors=frozenset(TensorStorage(tensor_renaming[ts.tensor_id],
                                        ts.backer_id,
                                        ts.above_loop_index,
                                        ts.tile_size)
                          for ts in tiling.tensors)
    )


def _convert_stats(from_einsum: int, to_einsum: int, stats, rank_renaming, tensor_renaming):
    stats = deepcopy(stats)
    for s in stats:
        s[MAPPING][to_einsum] = s[MAPPING].pop(from_einsum)
    return stats
    


def detect_similar_einsums(workload, analyzer, return_all_as_unique=False):
    if return_all_as_unique:
        return {ref: {} for ref in workload.einsum_id_to_name()}

    ref_to_to_einsums = {}
    for einsum in workload.einsum_id_to_name():
        found = False
        for from_einsum in ref_to_to_einsums:
            rank_renaming, tensor_renaming = is_equivalent(from_einsum,
                                                           einsum,
                                                           workload,
                                                           analyzer)
            if rank_renaming is not None:
                ref_to_to_einsums[from_einsum][einsum] = (rank_renaming,
                                                            tensor_renaming)
                found = True
                break
        if not found:
            ref_to_to_einsums[einsum] = {}
    return ref_to_to_einsums


def convert_rank_to_group_renaming(ref_to_to_einsums, equiv_ranks):
    return {
        ref: {
            other: (_convert_rank_renaming(rank_renaming, equiv_ranks),
                    tensor_renaming)
            for other, (rank_renaming, tensor_renaming) in others.items()
        }
        for ref, others in ref_to_to_einsums.items()
    }


def _convert_rank_renaming(rank_renaming, equiv_ranks):
    # The Tiling class uses string ids
    return {
        str(equiv_ranks.rank_to_group_id[r1])
        :
        str(equiv_ranks.rank_to_group_id[r2])
        for r1, r2 in rank_renaming.items()
    }
