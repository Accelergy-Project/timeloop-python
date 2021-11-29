from bindings import ArchProperties
from pytimeloop.config import Config
from pytimeloop.engine import Accelerator, AcceleratorPool
from pytimeloop.mapspace import MapSpace, Dimension
from pytimeloop.model import ArchSpecs, SparseOptimizationInfo
from pytimeloop.mapping import ArchConstraints, Mapping
from pytimeloop.problem import Workload
from pytimeloop.search import SearchAlgorithm, SearchStatus

from enum import Enum
from functools import reduce
import operator
import logging
import multiprocessing as mp


class Betterness(Enum):
    BETTER = 1
    SLIGHTLY_BETTER = 2
    WORSE = -1
    SLIGHTLY_WORSE = -2


class SearchTask:
    def __init__(self, task_id, mapping, only_bypass_changed):
        self.task_id = task_id
        self.mapping = mapping
        self.only_bypass = only_bypass_changed


class Mapper:
    """
    A mapper that finds optimal mapping given architecture, workload, mapspace,
    and search algorithms.

    The mapper runs a loop that:
      - consults the search algorithm for a mapping (via a call to
        `SearchAlgorithm.next`),
      - evaluates the mapping,
      - reports the result to the search algorithm.

    For better parallelism, since generating a mapping is often faster than
    evaluating it, a search algorithm may be asked for a mapping before a
    result is reported. A search algorithm can return None to signify it needs
    a report before sending the next mapping. The mapper will attempt to get
    a new mapping from the search algorithms until either all search algorithms
    returned None or the accelerator pool is busy.

    In general, set `accelerator_pool_num_threads` close to the maximum number
    of expected concurrent evaluations for best performance.

    Arguments:
        arch_specs: architecture specification.
        workload: workload specification.
        arch_constraints: mapping constraints due to architectural
            characteristics.
        mapspace: the mapspaces to be searched. Each mapspace will be used for
            the corresponding search algorithm, thus it has to be the same
            length as `search_algs`.
        search_algs: a list of search algorithms. Can be different algorithms.
        sparse_opts_info: sparse optimizations info.
        metrics: metrics to optimize. Defaults to ['edp'].
        accelerator_pool_num_threads: number of parallel workers in the
            accelerator pool used for evaluation. Defaults to number of CPU
            cores.
        search_size: number of mappings to search. Zero means search the whole
            mapspace. Defaults to 0.
        timeout: number of consecutive invalid mappings before giving up.
            Defaults to 1000.
        victory_condition: number of suboptimal mappings found before
            finishing. Defaults to 500.
        penalize_consecutive_bypass_fails: If True, bypass fail is counted as
            an invalid mapping. Defaults to False.
    """

    def __init__(self, arch_specs, workload, arch_constraints, mapspaces,
                 search_algs, sparse_opts_info, metrics=['edp'],
                 accelerator_pool_num_threads=mp.cpu_count(), search_size=0,
                 timeout=1000, victory_condition=500,
                 penalize_consecutive_bypass_fails=False):
        # Architecture configuration
        self.arch_specs = arch_specs
        self.arch_props = ArchProperties(self.arch_specs)

        # Problem configuration
        self.workload = workload

        # Mapper configuration
        self.num_threads = accelerator_pool_num_threads
        self.metrics = metrics
        self.timeout = timeout

        self.search_size = search_size
        if self.search_size > 0:
            self.search_size = 1 + (self.search_size - 1) / len(search_algs)

        self.victory_condition = 500
        self.penalize_consecutive_bypass_fails = \
            penalize_consecutive_bypass_fails

        # Architecture constraints
        self.constraints = arch_constraints

        # Mapspace configuration
        self.split_mapspaces = mapspaces

        # Search configuration
        self.search = search_algs

        # Sparse optimizations
        self.sparse_optimizations = sparse_opts_info

        # TODO: characterize workload on whether it has metadata

        # Data structures for each search algorithm
        self.total_maps = [0] * len(self.search)
        self.valid_maps = [0] * len(self.search)
        self.invld_maps_mapcnstr = [0] * len(self.search)
        self.invld_maps_eval = [0] * len(self.search)
        self.maps_since_last_best = [0] * len(self.search)
        self.prev_map_id = [None] * len(self.search)
        self.terminate = [False] * len(self.search)

    def run(self, log_level=logging.INFO):
        # Setup logger
        logger = logging.getLogger(__name__ + '.' + __class__.__name__)
        logger.setLevel(log_level)

        accelerator_pool = AcceleratorPool(self.arch_specs, self.num_threads)

        self.best_result = None
        # Maps search algorithm index to outstanding SearchTask
        outstanding_tasks = []
        for i in range(len(self.search)):
            outstanding_tasks.append(
                self._search_send_next(i, accelerator_pool))

        while True:
            if reduce(operator.and_, self.terminate):
                break
            result = accelerator_pool.get_result()
            for idx, search_task in enumerate(outstanding_tasks):
                if (not self.terminate[idx]
                        and search_task.task_id == result.id):
                    self._search_report(idx, result, search_task)
                    outstanding_tasks[idx] = self._search_send_next(
                        idx,
                        accelerator_pool)
                    break

        engine = Accelerator(self.arch_specs)
        eval_stat = engine.evaluate(self.best_mapping, self.workload,
                                    self.sparse_optimizations)
        return eval_stat, self.best_mapping

    def _search_send_next(self, i, accelerator_pool):
        if self.search_size > 0 and self.valid_maps[i] == self.search_size:
            self.terminate[i] = True

        if (self.victory_condition > 0 and
                self.maps_since_last_best[i] == self.victory_condition):
            self.terminate = True

        if (self.invld_maps_mapcnstr[i] +
                self.invld_maps_eval[i] > 0 and self.invld_maps_mapcnstr[i] +
                self.invld_maps_eval[i] == self.timeout):
            self.terminate = True

        # Get next mapping from search algorithm
        map_id = self.search[i].next()
        if map_id is None:
            self.terminate[i] = True

        if self.terminate[i]:
            return None

        # Check if the only change vs. the previous mapping was in the bypass
        # dimension. This is useful later.
        only_bypass_changed = False
        if self.total_maps[i] > 1:
            match = True
            for dim in Dimension.values():
                match = match and (map_id[dim] == self.prev_map_id[dim])
            only_bypass_changed = match
        self.prev_map_id = map_id

        # Begin mapping
        # Stage 1: make sure mapping is valid
        const_status, mapping = self.split_mapspaces[i].construct_mapping(
            map_id, True)
        success = reduce(lambda a, b: a and b.success, const_status)

        self.total_maps[i] += 1

        if not success:
            self.invalid_maps_mapcnstr[i] += 1
            # missing: diagnostics
            self.search[i].report(SearchStatus.MappingConstructionFailure)
            return None

        # Stage 2 & 3: pre-evaluation and evaluation
        task_id = accelerator_pool.evaluate(mapping, self.workload,
                                            self.sparse_optimizations, False)

        return SearchTask(task_id, mapping, only_bypass_changed)

    def _search_report(self, i, result, search_task):
        if result.eval_status is None:  # Failed in pre-evaluation
            if self.penalize_consecutive_bypass_fails\
                    or not search_task.only_bypass:
                self.invld_maps_eval[i] += 1
            self.search[i].report(SearchStatus.EvalFailure, 0)
            return

        # Evaluated
        success = reduce(lambda acc, x: acc and x.success, result.eval_status)
        if not success:
            if self.penalize_consecutive_bypass_fails\
                    or not search_task.only_bypass:
                self.invalid_maps_eval[i] += 1
            self.search[i].report(SearchStatus.EvalFailure)
            return

        # Success!
        self.valid_maps[i] += 1
        self.invld_maps_mapcnstr[i] = 0
        self.invld_maps_eval[i] = 0
        self.search[i].report(SearchStatus.Success,
                              Mapper._cost(result, self.metrics[0]))
        # missing: log_stats, log_suboptimal
        if Mapper._is_better(result, self.best_result, self.metrics):
            self.best_result = result
            self.best_mapping = search_task.mapping
            # missing: log_optimal and log_suboptimal

    @staticmethod
    def _is_better(candidate, incumbent, metrics):
        if incumbent is None:
            return True
        betterness = Mapper._is_better_recur(candidate, incumbent, metrics)
        return betterness.value > 0

    @staticmethod
    def _is_better_recur(candidate, incumbent, metrics):
        tolerance = 0.001

        candidate_cost = Mapper._cost(candidate, metrics[0])
        incumbent_cost = Mapper._cost(incumbent, metrics[0])

        if incumbent_cost == 0:
            relative_improvement = 1.0
        else:
            relative_improvement = ((incumbent_cost - candidate_cost)
                                    / incumbent_cost)

        if abs(relative_improvement) > tolerance:  # Clear winner
            if relative_improvement > 0:
                return Betterness.BETTER
            else:
                return Betterness.WORSE
        else:
            if len(metrics) == 1:
                if relative_improvement > 0:
                    return Betterness.SLIGHTLY_BETTER
                else:
                    return Betterness.SLIGHTLY_WORSE
            else:
                lsm = Mapper._is_better_recur(
                    candidate, incumbent, metrics[1:])

                if lsm == Betterness.BETTER or lsm == Betterness.WORSE:
                    return lsm
                elif relative_improvement > 0:
                    return Betterness.SLIGHTLY_BETTER
                else:
                    return Betterness.SLIGHTLY_WORSE

    @staticmethod
    def _cost(stats, metric):
        if metric == 'delay':
            return stats.cycles
        if metric == 'energy':
            return stats.energy
        if metric == 'last-level-accesses':
            return stats.last_level_accesses
        assert metric == 'edp'
        return stats.energy * stats.cycles
