from collections import defaultdict
from collections.abc import Mapping
import copy
import itertools
from math import exp
import random
import time

import pandas as pd
from joblib import delayed

from pytimeloop.looptree.equivalent_ranks import PairwiseEquivalentRanks

from pytimeloop.fastfusion.sim import SIM, Loop, TensorStorage, Tiling
from pytimeloop.fastfusion.pareto import MAPPING, Pareto, is_special_col, merge_cross
from pytimeloop.fastfusion.util import fzs, parallel, debugger_active

from pytimeloop.fastfusion.plot.looptree import tilings2looptree#, NotEnoughLoopsError,




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


def get_possible_translations(
    t: Tiling, 
    pairwise_equivalent_ranks: dict[str, set[str]],
    full_equivalent_ranks: dict[str, set[str]],
    right_ranks: set[str]
):
    # Fused ranks should be transitive, but if a fused loop indexes into two
    # different ranks in the next Einsum, we can't fuse becuase it will tile in
    # multiple directions.
    #
    # The first union checks what loops we CAN fuse with in the next Einsum. The
    # second union checks what loops MUST index into in the next
    #
    # Einsum. If we alias into multiple ranks, we can't fuse. Otherwise, try out
    # each possible rank.
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

class GroupOfSIMsHolder:
    def __init__(self, einsum_name: str, sim_list: list[SIM]):
        self.einsum_name: str = einsum_name
        self.sims: list[SIM] = sim_list
        self.tensor_names: set[str] = set(sim_list[0].tensor_names)

    def __getitem__(self, i):
        return self.sims[i]

class MapsapceGlobals:
    def __init__(self, sims: dict[str, list[SIM]], einsum2ranks: dict[str, set[str]], pairwise_equivalent_ranks: PairwiseEquivalentRanks):
        self.sims = sims
        self.einsum_names = list(sims.keys())
        self.storage2possible_loops_above = self._create_storage2possible_loops_above()
        self.tensor2storage = self._create_tensor2storage()
        self.rank_translations = self._create_rank_translations(einsum2ranks)
        self.full_equivalent_ranks = self._create_full_equivalent_ranks(pairwise_equivalent_ranks)
        self.tensor_names = set().union(*(s.tensor_names for s in sims.values()))
        
    def _create_storage2possible_loops_above(self):
        storage2possible_loops_above = {}
        for einsum_name, sim_list in self.items():
            storage2possible_loops_above[einsum_name] = defaultdict(set)
            for sim in sim_list:
                for storage in sim.tiling.storage:
                    storage2possible_loops_above[einsum_name][storage] |= set(sim.tiling.loops[:storage.above_loop_index])
        return {e: {s: list(l) for s, l in d.items()} for e, d in self.storage2possible_loops_above.items()}

    def _create_tensor2storage(self):
        tensor2storage = {}
        for t in self.tensor_names:
            possible_storage = []
            for einsum_name, sim_list in self.items():
                cur_storage = set()
                if t not in sim_list[0].tensor_names:
                    continue
                for sim in sim_list:
                    storage = sim.tiling.get_tensor_storage(t)
                    cur_storage.add(storage)
                possible_storage.append(cur_storage)
            tensor2storage[t] = list(set.intersection(*possible_storage))
        return tensor2storage
    
    def _create_rank_translations(self, einsum2ranks: dict[str, set[str]]):
        rank_translations = {}
        for einsum_name, ranks in einsum2ranks.items():
            translations = {einsum_name2: {} for einsum_name2 in self.einsum_names}
            for einsum_name2, ranks2 in einsum2ranks.items():
                for rank in ranks:
                    equiv = self.full_equivalent_ranks[rank] & ranks2
                    translations[einsum_name2][rank] = equiv
            rank_translations[einsum_name] = {k: {k2: list(v2) for k2, v2 in v.items()} for k, v in translations.items()}
        return rank_translations
    
    def _create_full_equivalent_ranks(self, pairwise_equivalent_ranks: PairwiseEquivalentRanks):
        full_equivalent_ranks = {k: set(v) for k, v in pairwise_equivalent_ranks.items()}
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
    

class Mapping:
    def __init__(self, sims: dict[str, list[SIM]]):
        self.sims = sims
        self.einsum_names = list(sims.keys())
        self.einsum2intra_choice = {einsum_name: 0 for einsum_name in self.einsum_names}
        self.einsum2tiling = {}
        for einsum_name, sim_list in sims.items():
            tensor_names = sim_list[0].tensor_names
            tensors = fzs(TensorStorage(t, 0, 0, 0) for t in tensor_names)
            self.einsum2tiling[einsum_name] = Tiling(tuple(), tensors)
            
    def fix_loops(self):
        for einsum, tiling in self.einsum2tiling.items():
            n_loops = max(s.above_loop_index for s in tiling.storage)
            
            # If there's too many loops then drop the extra ones
            if n_loops < len(tiling.loops):
                for e, t in self.einsum2tiling.items():
                    self.einsum2tiling[e] = t.update(loops=t.loops[:n_loops])
                    
            # If there's not enough loops then add some 
            if n_loops > len(tiling.loops):
                for tensor in tiling.storage:
                    for loop in range(len(tiling.loops), tensor.above_loop_index):
                        self.randomize_loop(tensor, loop, einsum)
                        self.force_loop_match(loop, einsum)

    def randomize_loop(self, storage: TensorStorage, index: int, einsum_name: str):
        if storage is None:
            storage = set().union(*(t.storage for t in self.einsum2tiling.values()))
            storage = random.choice(list(storage))
        
        
        tiling = self.einsum2tiling[einsum_name]
        candidates = storage2possible_loops_above[einsum_name][storage]
        loop = None
        if len(tiling.loops) <= index:
            choice = 'Randomizing'
        else:
            choice = random.choice(['Increasing', 'Decreasing', 'Randomizing'])

        if choice == 'Randomizing':
            new_loop = random.choice(candidates)
        else:
            loop = tiling.loops[index]
            rank, bound = loop.rank_name, loop.bound
            candidates = [c for c in candidates if c.rank_name == rank]
            if choice == 'Increasing':
                pruned_candidates = [c for c in candidates if c.bound > bound]
            else:
                pruned_candidates = [c for c in candidates if c.bound < bound]
            if len(pruned_candidates) == 0:
                choice = 'Randomizing'
            else:
                candidates = pruned_candidates
            if not candidates:
                return None

            new_loop = random.choice(candidates)
        print(f'{choice} loop {loop} -> {new_loop}')

        self.einsum2tiling[einsum_name] = tiling.set_loop(index, new_loop)
        return self.force_loop_match(index, einsum_name)


def fuse_sims(
    sims: dict[str, list[SIM]],
    pairwise_equivalent_ranks: PairwiseEquivalentRanks,
    einsum2ranks: dict[str, set[str]],
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
    lookahead_filter: bool = True,
):
    t0 = time.time()
    
    objective_function_col = None
    if objective_function_col is None:
        cols = [c for c in sims.values()][0][0].mapping.data.columns
        cols = [c for c in cols if not is_special_col(c)]
        assert len(cols) == 1
        objective_function_col = cols[0]

    full_equivalent_ranks = {k: set(v) for k, v in pairwise_equivalent_ranks.items()}
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
    
    rank_translations = {}
    for einsum_name, ranks in einsum2ranks.items():
        translations = {einsum_name2: {} for einsum_name2 in sims}
        for einsum_name2, ranks2 in einsum2ranks.items():
            for rank in ranks:
                equiv = full_equivalent_ranks[rank] & ranks2
                translations[einsum_name2][rank] = equiv
        rank_translations[einsum_name] = {k: {k2: list(v2) for k2, v2 in v.items()} for k, v in translations.items()}

    einsum_tiling_2_sims = {}
    for e, sim_list in sims.items():
        cur_sims = defaultdict(list)
        for sim in sim_list:
            cur_sims[sim.tiling].append(sim)
        einsum_tiling_2_sims[e] = {}
        for t, s in cur_sims.items():
            s = SIM.concat(s)
            if objective_function_col not in s.mapping.data.columns:
                s.mapping.data[objective_function_col] = 0
            s.mapping.data = s.mapping.data[[objective_function_col, MAPPING]]
            einsum_tiling_2_sims[e][t] = s

    # Implementing simulated annealing
    tensor_names = set()
    for einsum_name, sim_list in sims.items():
        tensor_names |= sim_list[0].tensor_names
    einsum_names = list(sims.keys())
    
    storage2possible_loops_above = {}
    for einsum_name, sim_list in sims.items():
        storage2possible_loops_above[einsum_name] = defaultdict(set)
        for sim in sim_list:
            for storage in sim.tiling.storage:
                storage2possible_loops_above[einsum_name][storage] |= set(sim.tiling.loops[:storage.above_loop_index])
    storage2possible_loops_above = {e: {s: list(l) for s, l in d.items()} for e, d in storage2possible_loops_above.items()}
    
    tensor2storage = {}
    for t in tensor_names:
        possible_storage = []
        for einsum_name, sim_list in sims.items():
            cur_storage = set()
            if t not in sim_list[0].tensor_names:
                continue
            for sim in sim_list:
                storage = sim.tiling.get_tensor_storage(t)
                cur_storage.add(storage)
            possible_storage.append(cur_storage)
        tensor2storage[t] = list(set.intersection(*possible_storage))
        
    # for tensor_name in tensor_names:
    #     possible_storage = set()
    #     for einsum_name, sim_list in sims.items():
    #         if tensor_name not in sim_list[0].tensor_names:
    #             continue
    #         for sim in sim_list:
    #             storage = sim.tiling.get_tensor_storage(tensor_name)
    #             possible_storage.add(storage)
    #             storage2possible_loops_above[storage] |= set(sim.tiling.loops[:storage.above_loop_index])
    #     if tensor_name not in tensor2storage:
    #         tensor2storage[tensor_name] = possible_storage
    #     else:
    #         possible_storage &= tensor2storage[tensor_name]
    # storage2possible_loops_above = {s: list(l) for s, l in storage2possible_loops_above.items()}

    einsum2tiling = {}
    for einsum_name, sim_list in sims.items():
        tensors = fzs(TensorStorage(t, 0, 0, 0) for t in sim_list[0].tensor_names)
        einsum2tiling[einsum_name] = Tiling(tuple(), tensors)
        
    einsum2intra_choice = {einsum_name: 0 for einsum_name in einsum_names}
        
    def switch_order(einsum2tiling, einsum2intra_choice):
        return einsum2tiling, einsum2intra_choice
        e0, e1 = random.sample(einsum_names, 2)
        print(f'Switching {e0} and {e1}')
        einsum2tiling[e0], einsum2tiling[e1] = einsum2tiling[e1], einsum2tiling[e0]
        return pad_loops(einsum2tiling, einsum2intra_choice)

    def move_memory(einsum2tiling, einsum2intra_choice):
        tensor = random.choice(list(tensor_names))
        storage = random.choice(tensor2storage[tensor])
        # If no change happened, skip
        for t in einsum2tiling.values():
            if storage in t.storage:
                return None
        print(f'Moving tensor {tensor} to storage {storage}')
        einsum2tiling = {e: t.set_tensor_storage(tensor, storage) for e, t in einsum2tiling.items()}
        return pad_loops(einsum2tiling, einsum2intra_choice)

    def pad_loops(einsum2tiling, einsum2intra_choice):
        print(f'\tPadding loops')
        for e, t in einsum2tiling.items():
            n_loops = max(s.above_loop_index for s in t.storage)
            if n_loops < len(t.loops):
                for e, t in einsum2tiling.items():
                    einsum2tiling[e] = t.update(loops=t.loops[:n_loops]) 
            if n_loops > len(t.loops):
                for s in t.storage:
                    for i in range(len(t.loops), s.above_loop_index):
                        randomize_loop(einsum2tiling, einsum2intra_choice, s, i, e)
                        force_loop_match(einsum2tiling, einsum2intra_choice, e, i)
                        break
        return einsum2tiling, einsum2intra_choice
    
    def force_loop_match(einsum2tiling, einsum2intra_choice, einsum_name, loop_index):
        print(f'\tForcing loop match for {einsum_name} at {loop_index}')
        tiling = einsum2tiling[einsum_name]
        for e, t in einsum2tiling.items():
            if e == einsum_name:
                continue
            t2 = einsum2tiling[e]
            shared_loop_index = max(t.shared_loop_index(t2.tensor_names), t2.shared_loop_index(t.tensor_names))
            for i in range(min(shared_loop_index, loop_index)+1):
                loop = tiling.loops[i]
                translations = rank_translations[einsum_name][e][loop.rank_name]
                if not translations:
                    return None
                rank_name = random.choice(translations)
                t2 = t2.set_loop(i, loop.update(rank_names=fzs((rank_name,))))
            einsum2tiling[e] = t2
        return einsum2tiling, einsum2intra_choice
        
    def randomize_loop(einsum2tiling, einsum2intra_choice, storage=None, index=None, einsum_name=None):
        print(f'Randomizing loop')
        if storage is None:
            storage = set().union(*(t.storage for t in einsum2tiling.values()))
            storage = random.choice(list(storage))
        if storage.above_loop_index == 0:
            return einsum2tiling, einsum2intra_choice
            
        if einsum_name is None:
            possible_einsums = [e for e, t in einsum2tiling.items() if storage in t.storage]
            assert possible_einsums
            einsum_name = random.choice(possible_einsums)
        tiling = einsum2tiling[einsum_name]

        candidates = storage2possible_loops_above[einsum_name][storage]
        if index is None:
            index = random.randint(0, storage.above_loop_index - 1)

        loop = None
        if len(tiling.loops) <= index:
            choice = 'Randomizing'
        else:
            choice = random.choice(['Increasing', 'Decreasing', 'Randomizing'])

        if choice == 'Randomizing':
            new_loop = random.choice(candidates)
        else:
            loop = tiling.loops[index]
            rank, bound = loop.rank_name, loop.bound
            candidates = [c for c in candidates if c.rank_name == rank]
            if choice == 'Increasing':
                pruned_candidates = [c for c in candidates if c.bound > bound]
            else:
                pruned_candidates = [c for c in candidates if c.bound < bound]
            if len(pruned_candidates) == 0:
                choice = 'Randomizing'
            else:
                candidates = pruned_candidates
            if not candidates:
                return None

            new_loop = random.choice(candidates)
        print(f'{choice} loop {loop} -> {new_loop}')

        einsum2tiling[einsum_name] = tiling.set_loop(index, new_loop)
        return force_loop_match(einsum2tiling, einsum2intra_choice, einsum_name, index)
    
    def eval_mapping(mapping):
        tree = tilings2looptree(mapping[MAPPING], None)
        reservations = tree.get_reservations()
        for resource, capacity in resource2capacity.items():
            if reservations[resource] > capacity:
                return float("inf")
        obj_val = mapping[objective_function_col]
        # print(f'Mapping has objective value {obj_val} and reservations {dict(reservations)}')
        return obj_val

    def evaluate(einsum2tiling, einsum2intra_choice, return_df=False):
        chosen_sims = []
        chosen_mappings = []
        for einsum_name, t in einsum2tiling.items():
            if t not in einsum_tiling_2_sims[einsum_name]:
                return float("inf")
            chosen_sims.append(einsum_tiling_2_sims[einsum_name][t])
            data = chosen_sims[-1].mapping.data
            data = data.iloc[einsum2intra_choice[einsum_name] % len(data)]
            chosen_mappings.append(data)

        mapping = {}
        for c in chosen_mappings:
            mapping.update(c[MAPPING])
        try:
            tree = tilings2looptree(mapping, None)
        except:# NotEnoughLoopsError:
            return float("inf")
        reservations = tree.get_reservations()
        for resource, capacity in resource2capacity.items():
            if reservations[resource] > capacity:
                return float("inf")
        if return_df:
            d = {
                objective_function_col: sum(c[objective_function_col] for c in chosen_mappings),
                MAPPING: mapping,
            }
            for k, v in reservations.items():
                d[f"RESOURCE_{k}_LEVEL_0"] = v
            return pd.DataFrame([d])
        
        score = sum(c[objective_function_col] for c in chosen_mappings)
        if score < 3.35e9:
            print(f'Found a mapping with score {score} after {time.time() - t0:.2f} seconds')
            assert False
        return score
            
        # N_SAMPLE = 10
        
        # def sample(data):
        #     return data.sample(N_SAMPLE) if len(data) > N_SAMPLE else data
            
        # data = sample(chosen_sims[0].mapping.data)
        for right in chosen_sims[1:]:
            data = sample(merge_cross(data, right.mapping.data, shared_loop_index=-1, live_tensors=set(), pareto_prune=False))
        return min(eval_mapping(row) for _, row in data.iterrows())

    def choose_intra_mapping(einsum2tiling, einsum2intra_choice):
        einsum_name = random.choice(einsum_names)
        intra_choice = random.randint(0, 1e9) # There won't be more than 1B intra-layer mappings
        print(f'Choosing intra-layer mapping {intra_choice} for {einsum_name}')
        einsum2intra_choice[einsum_name] = intra_choice
        return einsum2tiling, einsum2intra_choice

    mutations = [switch_order, move_memory, randomize_loop, choose_intra_mapping]

    def anneal(einsum2tiling, einsum2intra_choice, temperature, cooling_rate, n_iterations):
        prev_score = evaluate(einsum2tiling, einsum2intra_choice)
        for i in range(n_iterations):
            j = i + 1
            new_temp = temperature * (1 - j / n_iterations) / (1 + cooling_rate * j / n_iterations)
            new_einsum2tiling = copy.deepcopy(einsum2tiling)
            new_einsum2intra_choice = copy.deepcopy(einsum2intra_choice)
            mutated = random.choice(mutations)(new_einsum2tiling, new_einsum2intra_choice)
            if mutated is None:
                continue
            new_einsum2tiling, new_einsum2intra_choice = mutated
            new_score = evaluate(new_einsum2tiling, new_einsum2intra_choice)
            if new_score == float('inf'):
                continue
            
            if new_score <= prev_score or random.random() < exp((prev_score - new_score) / prev_score / new_temp):
                print(f'Iteration {i}: {prev_score} -> {new_score} accepted')
                prev_score = new_score
                einsum2tiling = new_einsum2tiling
                einsum2intra_choice = new_einsum2intra_choice
            else:
                print(f'Iteration {i}: {prev_score} -> {new_score} rejected')
            temperature *= 1 - cooling_rate
        return einsum2tiling, einsum2intra_choice
    
    # From SET code:
    # Cooling rate = 8
    # Temp = 0.07
    
    einsum2tiling, einsum2intra_choice = anneal(einsum2tiling, einsum2intra_choice, 0.07, 8, 1000000)
    
    
    # Just return the first one for now
    return evaluate(einsum2tiling, einsum2intra_choice, return_df=True)
    
    
