from collections import defaultdict
from collections.abc import Mapping
import copy
import itertools
from math import ceil, exp, prod
import math
import random
import threading
import time

import pandas as pd
from joblib import delayed

from pytimeloop.looptree.equivalent_ranks import PairwiseEquivalentRanks

from pytimeloop.fastfusion.sim import SIM, Loop, TensorStorage, Tiling
from pytimeloop.fastfusion.pareto import MAPPING, Pareto, is_special_col, VALID, col2nameloop
from pytimeloop.fastfusion.util import fzs, parallel, debugger_active

from pytimeloop.fastfusion.plot.looptree import (
    tilings2looptree,
)  # , NotEnoughLoopsError,

OBJECTIVE_COLUMN = None # None -> Product


def explore_fusion(
    einsum_to_result: Mapping,
    equivalent_ranks: PairwiseEquivalentRanks,
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
):
    return fuse_sims(
        mapping2sims(einsum_to_result),
        equivalent_ranks,
        resource2capacity,
        return_nmappings_nbuckets,
    )


def mapping2sims(einsum_to_result: Mapping):
    r = {}
    for einsum_name, compat_dict in einsum_to_result.items():
        r[einsum_name] = [paretofy(k, v) for k, v in compat_dict.items()]
    return list(r.values())


def paretofy(k, v):
    return SIM(k, Pareto(pd.DataFrame(v).fillna(0)))

class GroupOfSIMsHolder:
    def __init__(self, einsum_name: str, sim_list: list[SIM]):
        self.einsum_name: str = einsum_name
        self.sims: list[SIM] = sim_list
        self.tensor_names: set[str] = set(sim_list[0].tensor_names)

    def __getitem__(self, i):
        return self.sims[i]


class MapsapceGlobals:
    def __init__(
        self,
        sims: dict[str, list[SIM]],
        einsum2ranks: dict[str, set[str]],
        pairwise_equivalent_ranks: PairwiseEquivalentRanks,
        resource2capacity: dict,
        objective_function_cols: str,
    ):
        self.sims = sims
        self.einsum_names = list(sims.keys())
        self.tensor_names = set().union(*(s[0].tensor_names for s in sims.values()))
        self.resource2capacity = resource2capacity
        self.objective_function_cols = objective_function_cols
        self.storage2possible_loops_above = self._create_storage2possible_loops_above()
        self.storage2possible_loops_above_set = {
            k: {k2: set(v2) for k2, v2 in v.items()} for k, v in self.storage2possible_loops_above.items()
        }
        self.tensor2memories = self._create_tensor2memories()
        self.pairwise_equivalent_ranks = pairwise_equivalent_ranks
        self.full_equivalent_ranks = self._create_full_equivalent_ranks(
            pairwise_equivalent_ranks
        )
        self.einsum2ranks = einsum2ranks
        self.rank_translations = self._create_rank_translations(einsum2ranks)
        self.einsum_tiling_2_sim = self._create_einsum_tiling_2_sim()
        self.einsum_rank_index_to_loops = self._create_einsum_rank_index_to_loops()
        self.einsum2tensors = {k: set(s[0].tensor_names) for k, s in sims.items()}
        self.tiling2leftcompatibility, self.tiling2rightcompatibility, self.leftcompatibility2tiling, self.rightcompatibility2tiling = self._create_compatibility()
        self.einsum_tiling_2_valid, self.einsum_tiling_2_valid_porp = self._create_einsum_tiling_2_valid()
        
    def _create_einsum_tiling_2_valid(self):
        einsum_tiling_2_valid = {}
        einsum_tiling_2_valid_porp = {}
        for einsum_name, sim_list in self.sims.items():
            einsum_tiling_2_valid[einsum_name] = {}
            einsum_tiling_2_valid_porp[einsum_name] = {}
            for sim in sim_list:
                sim.mapping.data.reset_index(drop=True, inplace=True)
                valid_indices = list(sim.mapping.data.index[sim.mapping.data[VALID] == True])
                valid_indices_porp = len(valid_indices) / len(sim.mapping.data)
                einsum_tiling_2_valid[einsum_name][sim.tiling] = valid_indices
                einsum_tiling_2_valid_porp[einsum_name][sim.tiling] = valid_indices_porp
        return einsum_tiling_2_valid, einsum_tiling_2_valid_porp

        
    def get_live_tensors(self, *einsums: str):
        return set.union(*(self.einsum2tensors[e] for e in einsums))
        
    def _create_compatibility(self):
        tiling2leftcompatibility = {}
        tiling2rightcompatibility = {}
        def tilings2compatibility(tilings: list[Tiling], live_tensors: set[str], keep_tensors: set[str]):
            return {
                t: t.clear_dead_tensors(live_tensors=live_tensors, keep_tensors=keep_tensors)
                for t in tilings
            }
            
        for i, (einsum_name, sim_list) in enumerate(self.sims.items()):
            if i > 0:
                prev_live = self.get_live_tensors(*self.einsum_names[:i])
                prev = self.get_live_tensors(self.einsum_names[i - 1])
                tiling2leftcompatibility[einsum_name] = tilings2compatibility(
                    [s.tiling for s in sim_list],
                    prev_live,
                    prev,
                )
            if i < len(self.sims) - 1:
                next_live = self.get_live_tensors(*self.einsum_names[i + 1:])
                next = self.get_live_tensors(self.einsum_names[i])
                tiling2rightcompatibility[einsum_name] = tilings2compatibility(
                    [s.tiling for s in sim_list],
                    next_live,
                    next,
                )
        
        leftcompatibility2tiling = {}
        rightcompatibility2tiling = {}
        for einsum_name in self.einsum_names:
            for src, dst in (
                (tiling2leftcompatibility, leftcompatibility2tiling),
                (tiling2rightcompatibility, rightcompatibility2tiling),
            ):
                if einsum_name not in src:
                    continue
                dst = dst.setdefault(einsum_name, {})
                for k, v in src[einsum_name].items():
                    dst.setdefault(v, []).append(k)
        return (
            tiling2leftcompatibility,
            tiling2rightcompatibility,
            leftcompatibility2tiling,
            rightcompatibility2tiling,
        )
        
    def _create_einsum_tiling_2_sim(self):
        einsum_tiling_2_sim = {}
        for e, sim_list in self.sims.items():
            cur_sims = defaultdict(list)
            for sim in sim_list:
                cur_sims[sim.tiling].append(sim)
            einsum_tiling_2_sim[e] = {}
            for t, s in cur_sims.items():
                s = SIM.concat(s)
                einsum_tiling_2_sim[e][t] = s
        return einsum_tiling_2_sim
        
    def _create_storage2possible_loops_above(self):
        storage2possible_loops_above = {}
        for einsum_name, sim_list in self.sims.items():
            storage2possible_loops_above[einsum_name] = defaultdict(set)
            for sim in sim_list:
                for storage in sim.tiling.storage:
                    storage2possible_loops_above[einsum_name][storage] |= set(
                        sim.tiling.loops[: storage.above_loop_index]
                    )
        return {
            e: {s: list(l) for s, l in d.items()}
            for e, d in storage2possible_loops_above.items()
        }

    def _create_tensor2memories(self):
        tensor2memories = {}
        for t in self.tensor_names:
            possible_memories = []
            for einsum_name, sim_list in self.sims.items():
                cur_memories = set()
                if t not in sim_list[0].tensor_names:
                    continue
                for sim in sim_list:
                    storage = sim.tiling.get_tensor_storage(t)
                    cur_memories.add(storage)
                possible_memories.append(cur_memories)
            tensor2memories[t] = list(set.intersection(*possible_memories))
        return tensor2memories

    def _create_rank_translations(self, einsum2ranks: dict[str, set[str]]):
        rank_translations = {}
        for einsum_name, ranks in einsum2ranks.items():
            translations = {einsum_name2: {} for einsum_name2 in self.einsum_names}
            for einsum_name2, ranks2 in einsum2ranks.items():
                for rank in ranks:
                    equiv = self.full_equivalent_ranks[rank] & ranks2
                    translations[einsum_name2][rank] = equiv
            rank_translations[einsum_name] = {
                k: {k2: list(v2) for k2, v2 in v.items()}
                for k, v in translations.items()
            }
        return rank_translations

    def _create_full_equivalent_ranks(
        self, pairwise_equivalent_ranks: PairwiseEquivalentRanks
    ):
        full_equivalent_ranks = {
            k: set(v) for k, v in pairwise_equivalent_ranks.items()
        }
        changed = True
        while changed:
            changed = False
            for r in full_equivalent_ranks:
                for r2 in list(full_equivalent_ranks[r]):
                    for r3 in list(full_equivalent_ranks[r2]):
                        if r3 in full_equivalent_ranks[r]:
                            continue
                        changed = True
                        full_equivalent_ranks[r].add(r3)
        return full_equivalent_ranks
    
    def _create_einsum_rank_index_to_loops(self) -> dict[str, dict[str, dict[int, list[Loop]]]]:
        einsum_rank_index_to_loops = {}
        for einsum_name, sim_list in self.sims.items():
            einsum_rank_index_to_loops[einsum_name] = {}
            for sim in sim_list:
                for rank_index, loop in enumerate(sim.tiling.loops):
                    x = einsum_rank_index_to_loops[einsum_name].setdefault(loop.rank_name, {})
                    x.setdefault(rank_index, []).append(loop)
        return einsum_rank_index_to_loops

    def get_tensors(self, *einsums: str):
        return set.union(*(self.einsum2tensors[e] for e in einsums))


    def get_possible_translations(self, t: Tiling, to_einsum: str):
        pairwise_equivalent_ranks = self.pairwise_equivalent_ranks
        full_equivalent_ranks = self.full_equivalent_ranks
        right_ranks = self.einsum2ranks[to_einsum]
        def translate_loop(l: Loop):
            compatible_ranks = set.union(
                *(full_equivalent_ranks[n] for n in l.rank_names)
            ) & right_ranks
            pairwise_compatible_ranks = set.union(
                *(pairwise_equivalent_ranks[n] for n in l.rank_names)
            ) & right_ranks
            if len(pairwise_compatible_ranks) > 1:
                return
            for n in compatible_ranks:
                yield Loop(fzs((n,)), l.bound, l.is_spatial)
        for loops in itertools.product(*map(translate_loop, t.loops)):
            yield Tiling(loops, t.storage, t.tags)

class FailedMutation(Exception):
    pass


class Mapping:
    def __init__(self, sims: dict[str, list[SIM]]):
        self.einsum_names = list(sims.keys())
        self.einsum2intra_choice = {einsum_name: 0 for einsum_name in self.einsum_names}
        self.einsum2tiling = {}
        for einsum_name, sim_list in sims.items():
            tensor_names = sim_list[0].tensor_names
            tensors = fzs(TensorStorage(t, 0, 0, 0) for t in tensor_names)
            self.einsum2tiling[einsum_name] = Tiling(tuple(), tensors)
        self.prev_score = float("inf")
        # self.history = []
        class dummy_appender:
            def append(*args, **kwargs):
                pass
        self.history = dummy_appender()
        self.n_crossovers = 0
        self.n_mutations = 0
        
        self.n_changes = 0
        self.prev_eval_result = None
        self.prev_eval_at_n_changes = -1

    def fix_loops(self, mapspace_globals: MapsapceGlobals):
        """ Ensure that all tilings have the correct number of loops """
        self.n_changes += 1
        self.history.append("Fixing loops")

        try: 
            for einsum in self.einsum_names:
                tiling = self.einsum2tiling[einsum]
                n_loops = max(t.above_loop_index for t in tiling.storage)

                # If there's too many loops then drop the extra ones
                if n_loops < len(tiling.loops):
                    self.einsum2tiling[einsum] = tiling.update(loops=tiling.loops[:n_loops])

                # If there's not enough loops then add some
                if n_loops > len(tiling.loops):
                    for tensor in tiling.storage:
                        for loop in range(len(tiling.loops), tensor.above_loop_index):
                            self.mutate_loop(mapspace_globals, tensor, loop, einsum)
                            self.force_loop_match(mapspace_globals, loop, einsum)
                assert n_loops == len(self.einsum2tiling[einsum].loops)
                
                tiling = self.einsum2tiling[einsum]
                tensors = tiling.storage
                for i in range(len(tiling.loops)):
                    tensors = list(t for t in tensors if t.above_loop_index > i)
                    if not tensors:
                        continue
                    possible_loops = set.intersection(
                        *(mapspace_globals.storage2possible_loops_above_set[einsum][t] for t in tensors)
                    )
                    if not possible_loops:
                        raise FailedMutation(f"No possible loops above {i} for {einsum}")
                    if tiling.loops[i] not in possible_loops:
                        new_loop = random.choice(list(possible_loops))
                        self.history.append(f"Fixing loop {i} for {einsum} to {new_loop}")
                        tiling = tiling.set_loop(i, new_loop)
                self.einsum2tiling[einsum] = tiling

        except FailedMutation:
            self.history.append(f"Failed to fix loops")
            raise FailedMutation("Failed to fix loops")


    def match_loops(
        self, index: int, einsum_name: str, mapspace_globals: MapsapceGlobals
    ):
        """ Ensure that loops match across Einsums """
        self.n_changes += 1
        tiling = self.einsum2tiling[einsum_name]
        for einsum_name2, tiling2 in self.einsum2tiling.items():
            if einsum_name2 == einsum_name:
                continue
            shared_loop_index = max(
                tiling.shared_loop_index(tiling2.tensor_names),
                tiling2.shared_loop_index(tiling.tensor_names),
            )
            for i in range(min(shared_loop_index, index) + 1):
                # Translate loop from einsum_name to einsum_name2
                loop = tiling.loops[i]
                translations = mapspace_globals.rank_translations[einsum_name][
                    einsum_name2
                ][loop.rank_name]
                if not translations:
                    raise FailedMutation(
                        f"Failed to translate loop {loop} from {einsum_name} to {einsum_name2}"
                    )
                rank_name = random.choice(translations)
                tiling2 = tiling2.set_loop(i, loop.update(rank_names=fzs((rank_name,))))
            self.einsum2tiling[einsum_name2] = tiling2


    def mutate_loop(
        self,
        mapspace_globals: MapsapceGlobals,
        storage: TensorStorage=None,
        index: int=None,
        einsum_name: str=None,
    ):
        self.n_changes += 1
        if storage is None:
            memories = set().union(*(t.storage for t in self.einsum2tiling.values()))
            storage = random.choice(list(memories))
            if storage.above_loop_index == 0:
                raise FailedMutation(f"No loops above {storage} to mutate")
        if index is None:
            index = random.randint(0, storage.above_loop_index - 1)
        if einsum_name is None:
            possible_einsums = [e for e, t in self.einsum2tiling.items() if storage in t.storage]
            assert possible_einsums
            einsum_name = random.choice(possible_einsums)

        tiling = self.einsum2tiling[einsum_name]
        prev_loop = None

        choice = random.choice(["Increasing", "Decreasing", "Randomizing"])
        if len(tiling.loops) <= index:
            choice = "Randomizing"

        candidates = mapspace_globals.storage2possible_loops_above[einsum_name][storage]
        if choice == "Randomizing":
            new_loop = random.choice(candidates)
        else:
            prev_loop = tiling.loops[index]
            rank, bound = prev_loop.rank_name, prev_loop.bound
            comparison = lambda x, y: x > y if choice == "Increasing" else x < y

            candidates = [
                c
                for c in candidates
                if comparison(c.bound, bound) and c.rank_name == rank
            ]
            if not candidates:
                raise FailedMutation(
                    f"{choice} {prev_loop} for {einsum_name} at {index} failed"
                )
            new_loop = random.choice(candidates)

        self.history.append(f"{choice} loop {index} for {einsum_name} to {new_loop}")
        self.einsum2tiling[einsum_name] = tiling.set_loop(index, new_loop)

    def get_shared_loop_index(
            self, 
            mapspace_globals: MapsapceGlobals, 
            einsum_name0: int, 
            einsum_name1: int
        ):
        einsum_names = list(self.einsum2tiling.keys())
        if einsum_name0 == einsum_name1:
            einsum_name = einsum_names[einsum_index0]
            return len(self.einsum2tiling[einsum_name].loops) - 1
        
        einsum_index0 = einsum_names.index(einsum_name0)
        einsum_index1 = einsum_names.index(einsum_name1)
        
        if einsum_index0 > einsum_index1:
            einsum_index0, einsum_index1 = einsum_index1, einsum_index0
            
        tiling0 = self.einsum2tiling[einsum_names[einsum_index0]]
        tiling1 = self.einsum2tiling[einsum_names[einsum_index1]]
        left_tensors = mapspace_globals.get_tensors(*einsum_names[:einsum_index0 + 1])
        right_tensors = mapspace_globals.get_tensors(*einsum_names[einsum_index1:])
        return max(
            tiling0.shared_loop_index(right_tensors),
            tiling1.shared_loop_index(left_tensors),
        )

    def force_loop_match(
        self, mapspace_globals: MapsapceGlobals, index: int, einsum_name: str, 
    ):
        self.n_changes += 1
        tiling = self.einsum2tiling[einsum_name]
        for einsum_name2, tiling2 in self.einsum2tiling.items():
            if einsum_name2 == einsum_name:
                continue
            shared_loop_index = self.get_shared_loop_index(mapspace_globals, einsum_name, einsum_name2)
            rank_translations = mapspace_globals.rank_translations[einsum_name][einsum_name2]
            for i in range(min(shared_loop_index, index) + 1):
                loop = tiling.loops[i]
                translations = rank_translations[loop.rank_name]
                if not translations:
                    raise FailedMutation(
                        f"Failed to translate loop {loop} from {einsum_name} to {einsum_name2}"
                    )
                rank_name = random.choice(translations)
                tiling2 = tiling2.set_loop(i, loop.update(rank_names=fzs((rank_name,))))
            self.einsum2tiling[einsum_name2] = tiling2

    def mutate_backing_storage(self, mapspace_globals: MapsapceGlobals):
        self.n_changes += 1
        tensor = random.choice(list(mapspace_globals.tensor_names))
        storage = random.choice(mapspace_globals.tensor2memories[tensor])
        for t in self.einsum2tiling.values():
            if storage in t.storage:
                raise FailedMutation(
                    f"Moving tensor {tensor} to storage {storage} failed"
                )
        self.history.append(f"Moving tensor {tensor} to storage {storage}")
        for einsum, tiling in self.einsum2tiling.items():
            self.einsum2tiling[einsum] = tiling.set_tensor_storage(tensor, storage)
        self.fix_loops(mapspace_globals)

    def mutate_order(self, mapspace_globals: MapsapceGlobals):
        return
        self.n_changes += 1
        e0, e1 = random.sample(self.einsum_names, 2)
        print(f"Switching {e0} and {e1}")
        self.einsum2tiling[e0], self.einsum2tiling[e1] = (
            self.einsum2tiling[e1],
            self.einsum2tiling[e0],
        )
        self.fix_loops(mapspace_globals)

    def evaluate(self, mapspace_globals: MapsapceGlobals, return_df=False) -> float:
        if self.n_changes == self.prev_eval_at_n_changes and not return_df:
            return self.prev_eval_result, 1
        chosen_sims = []
        chosen_mappings = {}
        n_evaluations = 1
        
        if self.n_changes == self.prev_eval_at_n_changes and not return_df:
            return self.prev_eval_result, 1
        self.prev_eval_at_n_changes = self.n_changes
        self.prev_eval_result = float("inf")

        for einsum_name, t in self.einsum2tiling.items():
            if t not in mapspace_globals.einsum_tiling_2_sim[einsum_name]:
                assert not return_df
                return float("inf"), n_evaluations
            sim = mapspace_globals.einsum_tiling_2_sim[einsum_name][t]
            chosen_sims.append(sim)
            intra_mappings = sim.mapping.data
            mapping = intra_mappings.iloc[self.einsum2intra_choice[einsum_name] % len(intra_mappings)]
            if not mapping[VALID]:
                valid_indices = mapspace_globals.einsum_tiling_2_valid[einsum_name][t]
                valid_porp = mapspace_globals.einsum_tiling_2_valid_porp[einsum_name][t]
                if valid_porp == 0:
                    n_evaluations += len(mapping)
                    return float("inf"), n_evaluations
                choice = valid_indices[self.einsum2intra_choice[einsum_name] % len(valid_indices)]
                self.einsum2intra_choice[einsum_name] = choice
                n_evaluations += 1 / valid_porp
                mapping = intra_mappings.iloc[choice]
            assert mapping[VALID]
            
            # for i in range(10000): # Intra-layer search to find a valid mapping
            #     if VALID not in mapping or mapping[VALID]:
            #         break
            #     n_evaluations += 1
            #     self.einsum2intra_choice[einsum_name] = random.randint(1, 1e12)
            #     mapping = intra_mappings.iloc[self.einsum2intra_choice[einsum_name] % len(intra_mappings)]
            if VALID in mapping and not mapping[VALID]:
                assert not return_df
                return float("inf"), n_evaluations
            chosen_mappings[einsum_name] = mapping

        # mapping = {}
        # for c in chosen_mappings:
        #     mapping.update(c[MAPPING])
        try:
            # tree = tilings2looptree(mapping, None)
            tree = tilings2looptree(
                self.einsum2tiling,
                add_reservations=chosen_mappings,
            )
            # tree.validate_loops(mapspace_globals.einsum2ranks)
        except:
            assert not return_df
            return float("inf"), n_evaluations

        reservations = tree.get_reservations()
        for resource, capacity in mapspace_globals.resource2capacity.items():
            if capacity is not None and reservations.get(resource, 0) > capacity:
                assert not return_df
                return float("inf"), n_evaluations
            
        obj_cols = mapspace_globals.objective_function_cols
        score = prod(sum(c[col] for c in chosen_mappings.values()) for col in obj_cols)
        if return_df:
            d = {col: sum(c[col] for c in chosen_mappings.values()) for col in obj_cols}
            d[MAPPING] = mapping
            for k, v in reservations.items():
                d[f"RESOURCE_{k}_LEVEL_0"] = v
            self.prev_eval_result = score
            return pd.DataFrame([d]), n_evaluations
        self.prev_eval_result = score
        return score, n_evaluations
    
    def mutate_intra_mapping(self, mapspace_globals: MapsapceGlobals):
        self.n_changes += 1
        einsum_name = random.choice(self.einsum_names)
        intra_choice = random.randint(0, 1e12)
        self.history.append(f"Choosing intra-layer mapping {intra_choice} for {einsum_name}")
        self.einsum2intra_choice[einsum_name] = intra_choice
    
    def get_mutation_functions(self):
        return [self.mutate_loop, self.mutate_backing_storage, self.mutate_order, self.mutate_intra_mapping]

    def crossover(self, other: Mapping, mapspace_globals: MapsapceGlobals):
        child = copy.deepcopy(other)
        einsum_name = random.choice(child.einsum_names)
        try:
            child.einsum2tiling[einsum_name] = self.einsum2tiling[einsum_name]
            child.einsum2intra_choice[einsum_name] = self.einsum2intra_choice[einsum_name]
            child.n_changes += 1
            for i in range(len(child.einsum2tiling[einsum_name].loops)):
                child.match_loops(i, einsum_name, mapspace_globals)
            child.fix_loops(mapspace_globals)
            child.n_crossovers += 1
        except FailedMutation:
            return copy.deepcopy(other)
        return child
    
    @staticmethod
    def create_random_mapping(mapspace_globals: MapsapceGlobals):
        mapping = Mapping(mapspace_globals.sims)
        prev_compatibility: Tiling = None
        einsum_names = list(mapping.einsum2tiling.keys())
        for i, einsum_name in enumerate(einsum_names):
            sim_list = mapspace_globals.sims[einsum_name]
            if prev_compatibility is None:
                sim = random.choice(sim_list)
                mapping.einsum2tiling[einsum_name] = sim.tiling
                if len(einsum_names) == 1:
                    break
                prev_compatibility = mapspace_globals.tiling2rightcompatibility[einsum_name][sim.tiling]
                live_tensors = mapspace_globals.get_live_tensors(*einsum_names[i+1:])
                prev_compatibility = prev_compatibility.clear_dead_tensors(live_tensors=live_tensors)
                continue

            tilings = []
            compatiblity_options = mapspace_globals.leftcompatibility2tiling[einsum_name]
            cur_tensors = mapspace_globals.get_tensors(einsum_name)
            for translation in mapspace_globals.get_possible_translations(
                prev_compatibility,
                einsum_name
            ):
                translation = translation.clear_dead_tensors(live_tensors=cur_tensors, keep_loops=True)
                if translation in compatiblity_options:
                    tilings.extend(compatiblity_options[translation])
            if not tilings:
                raise FailedMutation(f"No tilings for {einsum_name} with {prev_compatibility}")
            tiling = random.choice(tilings)
            sim = mapspace_globals.einsum_tiling_2_sim[einsum_name][tiling]
            mapping.einsum2tiling[einsum_name] = tiling
            mapping.einsum2intra_choice[einsum_name] = random.randint(0, 1e12)
            if i == len(einsum_names) - 1:
                break
            
            new_compatibility: Tiling = mapspace_globals.tiling2rightcompatibility[einsum_name][tiling]

            # Combine prev_compatibility and new_compatibility
            live_tensors = mapspace_globals.get_live_tensors(*einsum_names[i+1:])
            prev_compatibility = prev_compatibility.merge_next(new_compatibility, live_tensors)
        return mapping
    
def get_accept_function(temperature, cooling_rate, evaluations_tracker):
    proportion = evaluations_tracker.evaluations / evaluations_tracker.max_evaluations
    new_temp = (
        temperature
        * (1 - proportion)
        / (1 + cooling_rate * proportion)
    )
    def accept(prev_score, new_score):
        if new_score == float("inf"):
            return False
        if new_score <= prev_score:
            return True
        scaleby = prev_score * new_temp
        if scaleby > 0 and random.random() < exp((prev_score - new_score) / scaleby):
            return True
        return False
    return accept

def mutate(mapping: Mapping, mapspace_globals: MapsapceGlobals, accept_function: callable):
    prev_mapping = copy.deepcopy(mapping)
    prev_score = mapping.prev_score
    n_evaluations = 1
    try:
        choice = random.choice(mapping.get_mutation_functions())
        choice(mapspace_globals)
    except FailedMutation:
        return prev_mapping, n_evaluations
    prev_score = mapping.prev_score
    new_score, n_evaluations = mapping.evaluate(mapspace_globals)
    if new_score == float("inf"):
        return prev_mapping, n_evaluations
    if accept_function(prev_score, new_score):
        return mapping, n_evaluations
    return prev_mapping, n_evaluations

def _fuse_sims(
    sims: dict[str, list[SIM]],
    mapspace_globals: MapsapceGlobals,
    n_threads: int,
    evaluations_tracker,
    algorithm: str
):
    random.seed(time.time() + hash(threading.get_ident()))  # Seed with thread ID
    evaluations_tracker.set_scale_by(len(mapspace_globals.einsum_names))
    evaluations_tracker.print_period *= n_threads
    evaluations_tracker.max_evaluations //= n_threads
    def anneal_population(population, mapspace_globals: MapsapceGlobals, n_rounds):
        temperature = 0.07
        cooling_rate = 8
        for i in range(n_rounds):
            accept_function = get_accept_function(temperature, cooling_rate, evaluations_tracker)
            # population = parallel([delayed(mutate)(m, mapspace_globals, accept_function) for m in population])
            for j, mapping in enumerate(population):
                population[j], evaluations = mutate(mapping, mapspace_globals, accept_function)
                if evaluations_tracker.add_evaluation(evaluations, population[j].prev_score):
                    return population
    
    def genetic_algorithm_population(population, mapspace_globals: MapsapceGlobals, n_rounds):
        population_size = len(population)
        crossover_rate = 0.7
        mutation_rate = 0.2

        def crossover(parent1: Mapping, parent2: Mapping):
            if random.random() > crossover_rate:
                return copy.deepcopy(parent1)
            return parent1.crossover(parent2, mapspace_globals)

        def mutate_individual(individual):
            individual = copy.deepcopy(individual)
            prev_mapping = copy.deepcopy(individual)
            if random.random() > mutation_rate:
                return individual
            try:
                mutation_function = random.choice(individual.get_mutation_functions())
                mutation_function(mapspace_globals)
                individual.n_mutations += 1
                return individual
            except FailedMutation:
                return prev_mapping

        best_fitness = float("inf")
        for generation in range(n_rounds):
            # Evaluate fitness
            fitness = [0] * len(population)
            for i, individual in enumerate(population):
                f, evaluations = individual.evaluate(mapspace_globals)
                fitness[i] = f
                best_fitness = min(best_fitness, f)
                if evaluations_tracker.add_evaluation(evaluations, best_fitness):
                    return population

            best_score = min(fitness)
            best_mapping = population[fitness.index(best_score)]

            # Selection (roulette wheel selection)
            total_fitness = sum(1.0 / (f + 1e-9) for f in fitness)
            probabilities = [(1.0 / (f + 1e-9)) / total_fitness for f in fitness]
            selected_indices = random.choices(range(len(population)), probabilities, k=population_size)

            # Crossover
            new_population = list(population[i] for i in selected_indices)
            for i in range(0, population_size, 2):
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[(i + 1) % population_size]]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.extend([child1, child2])

            # Mutation
            for i, individual in enumerate(new_population):
                new_population[i] = mutate_individual(individual)

            new_population.append(best_mapping) # Keep the best mapping around
            population = new_population

        return population
    
    def random_sample_population(population, mapspace_globals: MapsapceGlobals, n_rounds, prune=False):
        best_mapping = population[0]
        for i in range(n_rounds):
            try:
                mapping = Mapping.create_random_mapping(mapspace_globals)
            except FailedMutation:
                if not prune:
                    if evaluations_tracker.add_evaluation(1, float("inf")):
                        return [best_mapping]
                continue
            score, evaluations = mapping.evaluate(mapspace_globals)
            if evaluations_tracker.add_evaluation(evaluations, score):
                return [best_mapping]
        return [best_mapping]

    extra_args = {}
    if algorithm == "genetic":
        population_size = 1000 // n_threads
        callfunc = genetic_algorithm_population
    elif algorithm == "simulated_anneal":
        population_size = 100 // n_threads
        callfunc = anneal_population
    elif "random" in algorithm:
        population_size = 1
        callfunc = random_sample_population
        extra_args["prune"] = "pruned" in algorithm
        
    # Randomly intialize the population
    def get_random_mapping():
        while True:
            try:
                mapping = Mapping.create_random_mapping(mapspace_globals)
                score, evaluations = mapping.evaluate(mapspace_globals)
                evaluations_tracker.add_evaluation(evaluations, score)
                if score == float("inf"):
                    raise FailedMutation("Random mapping failed")
                return mapping
            except FailedMutation:
                pass
            
    population = [get_random_mapping() for _ in range(population_size)]

    n_rounds = 9999999999999999999999999
    results = callfunc(population, mapspace_globals, n_rounds)
    eval_results = []
    for m in results:
        try:
            eval_results.append(m.evaluate(mapspace_globals, return_df=True)[0])
        except:
            pass
    return pd.concat(eval_results), evaluations_tracker
    # pops, score_evaluations = zip(*results)
    # aggregate_score = []
    # aggregate_evaluations = []
    # for se in score_evaluations:
    #     if not aggregate_score:
    #         aggregate_score = [s for s, _ in se]
    #         aggregate_evaluations = [e for _, e in se]
    #     else:
    #         for i, (s, e) in enumerate(se):
    #             aggregate_score[i] = min(aggregate_score[i], s)
    #             aggregate_evaluations[i] += e

    # zipped = list(zip(aggregate_score, aggregate_evaluations))
    # print(f'Evaluations, Score')
    # for i in range(0, len(zipped), len(zipped) // 10):
    #     score, evaluations = zipped[i]
    #     print(f"{evaluations}, {score}")

    # mappings = list(itertools.chain(*pops))
    # mappings = pd.concat([m.evaluate(mapspace_globals, return_df=True)[0] for m in mappings])
    # mappings.sort_values(by=mapspace_globals.objective_function_cols, inplace=True)
    # return mappings, evaluations_tracker

def fuse_sims(
    sims: dict[str, list[SIM]],
    pairwise_equivalent_ranks: PairwiseEquivalentRanks,
    einsum2ranks: dict[str, set[str]],
    evaluations_tracker,
    algorithm: str,
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
    lookahead_filter: bool = False,
):

    objective_function_cols = None
    cols = next(iter(sims.values()))[0].mapping.data.columns
    if objective_function_cols is None:
        objective_function_cols = [c for c in cols if not is_special_col(c)]
    keepcols = []
    if MAPPING in cols:
        keepcols.append(MAPPING)
    if VALID in cols:
        keepcols.append(VALID)

    def detuplefy(s):
        s.mapping.detuplefy_data()
        return s

    for sim_list in sims.values():
        for sim in sim_list:
            for col in objective_function_cols:
                if col not in sim.mapping.data.columns:
                    sim.mapping.data[col] = 0
            reservations = [c for c in sim.mapping.data.columns if col2nameloop(c) is not None]
            sim.mapping.data = sim.mapping.data[objective_function_cols + keepcols + reservations]
            sim.mapping.detuplefy_data()

    mapspace_globals = MapsapceGlobals(
        sims,
        einsum2ranks,
        pairwise_equivalent_ranks,
        resource2capacity,
        objective_function_cols,
    )
    
    n_threads = 32
    
    while n_threads >= 1:
        try:
            results_and_trackers = parallel([delayed(_fuse_sims)(
                sims,
                mapspace_globals,
                n_threads=n_threads,
                evaluations_tracker=copy.deepcopy(evaluations_tracker),
                algorithm=algorithm,
            ) for _ in range(n_threads)], n_jobs=n_threads)
            results = pd.concat([r[0] for r in results_and_trackers])
            for t in results_and_trackers:
                evaluations_tracker.merge_with(t[1])
            return results
        except OSError as e:
            if n_threads == 1:
                raise OSError("Failed to fuse sims with 1 thread") from e
            print(f"Failed to fuse sims with {n_threads} threads, trying with {n_threads // 2}")
            n_threads //= 2
            
def fuse_sims_simulated_anneal(*args, **kwargs):
    return fuse_sims(*args, **kwargs, algorithm="simulated_anneal")
def fuse_sims_genetic(*args, **kwargs):
    return fuse_sims(*args, **kwargs, algorithm="genetic")
def fuse_sims_random(*args, **kwargs):
    return fuse_sims(*args, **kwargs, algorithm="random")