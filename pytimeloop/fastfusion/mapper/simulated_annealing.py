from collections import defaultdict
from collections.abc import Mapping
import copy
import itertools
from math import ceil, exp, prod
import math
import random
import time

import pandas as pd
from joblib import delayed

from pytimeloop.looptree.equivalent_ranks import PairwiseEquivalentRanks

from pytimeloop.fastfusion.sim import SIM, Loop, TensorStorage, Tiling
from pytimeloop.fastfusion.pareto import MAPPING, Pareto, is_special_col, VALID
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
        self.memory2possible_loops_above = self._create_memory2possible_loops_above()
        self.memory2possible_loops_above_set = {
            k: {k2: set(v2) for k2, v2 in v.items()} for k, v in self.memory2possible_loops_above.items()
        }
        self.tensor2memories = self._create_tensor2memories()
        self.full_equivalent_ranks = self._create_full_equivalent_ranks(
            pairwise_equivalent_ranks
        )
        self.rank_translations = self._create_rank_translations(einsum2ranks)
        self.einsum_tiling_2_sims = self._create_einsum_tiling_2_sims()
        self.einsum_rank_index_to_loops = self._create_einsum_rank_index_to_loops()
        self.einsum2tensors = {k: set(s[0].tensor_names) for k, s in sims.items()}
        
    def _create_einsum_tiling_2_sims(self):
        einsum_tiling_2_sims = {}
        for e, sim_list in self.sims.items():
            cur_sims = defaultdict(list)
            for sim in sim_list:
                cur_sims[sim.tiling].append(sim)
            einsum_tiling_2_sims[e] = {}
            for t, s in cur_sims.items():
                s = SIM.concat(s)
                einsum_tiling_2_sims[e][t] = s
        return einsum_tiling_2_sims
        
    def _create_memory2possible_loops_above(self):
        memory2possible_loops_above = {}
        for einsum_name, sim_list in self.sims.items():
            memory2possible_loops_above[einsum_name] = defaultdict(set)
            for sim in sim_list:
                for memory in sim.tiling.tensors:
                    memory2possible_loops_above[einsum_name][memory] |= set(
                        sim.tiling.loops[: memory.above_loop_index]
                    )
        return {
            e: {s: list(l) for s, l in d.items()}
            for e, d in memory2possible_loops_above.items()
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
                    memory = sim.tiling.get_tensor_storage(t)
                    cur_memories.add(memory)
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
        self.history = []

    def fix_loops(self, mapspace_globals: MapsapceGlobals):
        """ Ensure that all tilings have the correct number of loops """
        self.history.append("Fixing loops")
        try: 
            for einsum in self.einsum_names:
                tiling = self.einsum2tiling[einsum]
                n_loops = max(t.above_loop_index for t in tiling.tensors)

                # If there's too many loops then drop the extra ones
                if n_loops < len(tiling.loops):
                    self.einsum2tiling[einsum] = tiling.update(loops=tiling.loops[:n_loops])

                # If there's not enough loops then add some
                if n_loops > len(tiling.loops):
                    for tensor in tiling.tensors:
                        for loop in range(len(tiling.loops), tensor.above_loop_index):
                            self.mutate_loop(mapspace_globals, tensor, loop, einsum)
                            self.force_loop_match(mapspace_globals, loop, einsum)
                assert n_loops == len(self.einsum2tiling[einsum].loops)
                
                tiling = self.einsum2tiling[einsum]
                tensors = tiling.tensors
                for i in range(len(tiling.loops)):
                    tensors = list(t for t in tensors if t.above_loop_index > i)
                    if not tensors:
                        continue
                    possible_loops = set.intersection(
                        *(mapspace_globals.memory2possible_loops_above_set[einsum][t] for t in tensors)
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
        memory: TensorStorage=None,
        index: int=None,
        einsum_name: str=None,
    ):
        if memory is None:
            memories = set().union(*(t.tensors for t in self.einsum2tiling.values()))
            memory = random.choice(list(memories))
            if memory.above_loop_index == 0:
                raise FailedMutation(f"No loops above {memory} to mutate")
        if index is None:
            index = random.randint(0, memory.above_loop_index - 1)
        if einsum_name is None:
            possible_einsums = [e for e, t in self.einsum2tiling.items() if memory in t.tensors]
            assert possible_einsums
            einsum_name = random.choice(possible_einsums)

        tiling = self.einsum2tiling[einsum_name]
        prev_loop = None

        choice = random.choice(["Increasing", "Decreasing", "Randomizing"])
        if len(tiling.loops) <= index:
            choice = "Randomizing"

        candidates = mapspace_globals.memory2possible_loops_above[einsum_name][memory]
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

    def mutate_backing_memory(self, mapspace_globals: MapsapceGlobals):
        tensor = random.choice(list(mapspace_globals.tensor_names))
        memory = random.choice(mapspace_globals.tensor2memories[tensor])
        for t in self.einsum2tiling.values():
            if memory in t.tensors:
                raise FailedMutation(
                    f"Moving tensor {tensor} to memory {memory} failed"
                )
        self.history.append(f"Moving tensor {tensor} to memory {memory}")
        for einsum, tiling in self.einsum2tiling.items():
            self.einsum2tiling[einsum] = tiling.set_tensor_storage(tensor, memory)
        self.fix_loops(mapspace_globals)

    def mutate_order(self, mapspace_globals: MapsapceGlobals):
        return
        e0, e1 = random.sample(self.einsum_names, 2)
        print(f"Switching {e0} and {e1}")
        self.einsum2tiling[e0], self.einsum2tiling[e1] = (
            self.einsum2tiling[e1],
            self.einsum2tiling[e0],
        )
        self.fix_loops(mapspace_globals)

    def evaluate(self, mapspace_globals: MapsapceGlobals, return_df=False) -> float:
        chosen_sims = []
        chosen_mappings = []
        n_evaluations = 1
        for einsum_name, t in self.einsum2tiling.items():
            if t not in mapspace_globals.einsum_tiling_2_sims[einsum_name]:
                return float("inf"), n_evaluations
            chosen_sims.append(mapspace_globals.einsum_tiling_2_sims[einsum_name][t])
            intra_mappings = chosen_sims[-1].mapping.data
            mapping = intra_mappings.iloc[self.einsum2intra_choice[einsum_name] % len(intra_mappings)]
            for i in range(10000): # Intra-layer search to find a valid mapping
                if VALID not in mapping or mapping[VALID]:
                    break
                n_evaluations += 1
                self.einsum2intra_choice[einsum_name] = random.randint(1, 1e12)
                mapping = intra_mappings.iloc[self.einsum2intra_choice[einsum_name] % len(intra_mappings)]
            if VALID in mapping and not mapping[VALID]:
                return float("inf"), n_evaluations
            chosen_mappings.append(mapping)

        mapping = {}
        for c in chosen_mappings:
            mapping.update(c[MAPPING])
        try:
            tree = tilings2looptree(mapping, None)
            # tree.validate_loops(mapspace_globals.einsum2ranks)
        except:
            return float("inf"), n_evaluations

        reservations = tree.get_reservations()
        for resource, capacity in mapspace_globals.resource2capacity.items():
            if capacity is not None and reservations[resource] > capacity:
                return float("inf"), n_evaluations
            
        obj_cols = mapspace_globals.objective_function_cols
        self.prev_score = prod(sum(c[col] for c in chosen_mappings) for col in obj_cols)
        if return_df:
            d = {col: sum(c[col] for c in chosen_mappings) for col in obj_cols}
            d[MAPPING] = mapping
            for k, v in reservations.items():
                d[f"RESOURCE_{k}_LEVEL_0"] = v
            return pd.DataFrame([d]), n_evaluations
        return self.prev_score, n_evaluations
    
    def mutate_intra_mapping(self, mapspace_globals: MapsapceGlobals):
        einsum_name = random.choice(self.einsum_names)
        intra_choice = random.randint(0, 1e12)
        self.history.append(f"Choosing intra-layer mapping {intra_choice} for {einsum_name}")
        self.einsum2intra_choice[einsum_name] = intra_choice
    
    def get_mutation_functions(self):
        return [self.mutate_loop, self.mutate_backing_memory, self.mutate_order, self.mutate_intra_mapping]

    def mcts_loops_intra(self, mapspace_globals: MapsapceGlobals):
        # For each one, generate a setter function and a list of
        # options
        # Create a MCTS function using the setter and options
        
        # Assemble a list of matched loops
        einsum_names = list(self.einsum2tiling.keys())
        grouped_loops = {}
        for i, einsum_name in enumerate(self.einsum2tiling):
            tiling = self.einsum2tiling[einsum_name]
            for j in range(len(tiling.loops)):
                key = (einsum_name, j)
                grouped_loops.setdefault(key, []).append(key)
            if i == len(einsum_names) - 1:
                continue
            next_einsum_name = einsum_names[i + 1]
            shared_loop_index = self.get_shared_loop_index(
                mapspace_globals, einsum_name, next_einsum_name
            )
            for j in range(shared_loop_index + 1):
                key = (einsum_name, j)
                key2 = (next_einsum_name, j)
                grouped_loops[key2] = grouped_loops[key]
                del grouped_loops[key]
                
                
        # Generate a list of options for each loop
        options = {}
        for einsum_name, index in grouped_loops:
            rank = self.einsum2tiling[einsum_name].loops[index].rank_name
            options[(einsum_name, index)] = [
                r.bound for r in mapspace_globals.einsum_rank_index_to_loops[einsum_name][rank][index]
            ]
            
        N_ITERATIONS = 1000
            
#  auto cur_score = child->ave_reward + C * std::sqrt(std::log(n_visit) / child->n_visit);
            
        # MCTS implementation
        class MCTSNode:
            def __init__(self, parent=None, action=None):
                self.parent = parent
                self.action = action  # (einsum_name, index, choice)
                self.children = {}
                self.visits = 0
                self.value = 0
                self.unexplored_actions = []
                
            def is_fully_expanded(self):
                return len(self.unexplored_actions) == 0
                
            def select_child(self, c=1.414):
                """Select child using UCB1 formula"""
                return max(self.children.items(),
                           key=lambda item: item[1].value / (item[1].visits or 1) +
                           c * math.log((2 * (self.visits or 1))) ** 0.5 / (item[1].visits or 1))
                
            def expand(self):
                action = self.unexplored_actions.pop()
                self.children[action] = MCTSNode(parent=self, action=action)
                return self.children[action]
                
            def update(self, reward):
                self.visits += 1
                self.value += reward
        
        def mcts_search(mapping, mapspace_globals, iterations=N_ITERATIONS):
            # Create a copy of the mapping to work with
            best_mapping = copy.deepcopy(mapping)
            best_score, n_evaluations = mapping.evaluate(mapspace_globals)
            
            # Setup all possible actions
            all_actions = []
            for key, choice_list in options.items():
                einsum_name, index = key
                for choice in choice_list:
                    all_actions.append((einsum_name, index, choice))
            
            # Also add intra-mapping choices
            for einsum_name in mapping.einsum_names:
                for i in range(10):  # Add some reasonable number of intra-mapping choices
                    all_actions.append((einsum_name, "intra", i))
            
            # Define apply_action function
            def apply_action(m, action):
                if action[1] == "intra":
                    einsum_name, _, choice = action
                    m.einsum2intra_choice[einsum_name] = choice
                else:
                    einsum_name, index, choice = action
                    # Use the setter function to update the loop bound
                    for einsum_name2, index2 in grouped_loops[(einsum_name, index)]:
                        tiling = m.einsum2tiling[einsum_name2]
                        loop = tiling.loops[index2]
                        m.einsum2tiling[einsum_name2] = tiling.set_loop(index2, loop.update(bound=choice))
                return m
            
            # Define rollout function
            def rollout(m, depth=5):
                m_copy = copy.deepcopy(m)
                for _ in range(depth):
                    action = random.choice(all_actions)
                    try:
                        m_copy = apply_action(m_copy, action)
                    except Exception:
                        pass
                
                try:
                    score = m_copy.evaluate(mapspace_globals)
                    if score == float("inf"):
                        return 0  # Invalid mapping
                    return 1.0 / score  # Smaller score is better
                except:
                    return 0
            
            # MCTS main loop
            root = MCTSNode()
            root.unexplored_actions = list(all_actions)
            
            for _ in range(iterations):
                # Selection
                node = root
                mapping_copy = copy.deepcopy(mapping)
                
                while not node.is_fully_expanded() and node.children:
                    action, node = node.select_child()
                    mapping_copy = apply_action(mapping_copy, action)
                
                # Expansion
                if not node.is_fully_expanded():
                    node = node.expand()
                    mapping_copy = apply_action(mapping_copy, node.action)
                
                # Simulation
                reward = rollout(mapping_copy)
                
                # Backpropagation
                while node:
                    node.update(reward)
                    node = node.parent
                
                # Check if we found a better mapping
                try:
                    score = mapping_copy.evaluate(mapspace_globals)
                    if score < best_score:
                        best_score = score
                        best_mapping = copy.deepcopy(mapping_copy)
                except:
                    pass
            
            return best_mapping
            
        # Run MCTS
        self = mcts_search(self, mapspace_globals)
        return self

def fuse_sims_simulated_anneal(
    sims: dict[str, list[SIM]],
    mapspace_globals: MapsapceGlobals,
):
    t0 = time.time()
    
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
    
    def get_accept_function(iteration, temperature, cooling_rate):
        proportion = iteration / n_rounds
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
            if random.random() < exp((prev_score - new_score) / prev_score / new_temp):
                return True
            return False
        return accept
    
    t0 = time.time()
    
    def anneal_population(thread, population, mapspace_globals: MapsapceGlobals, temperature, cooling_rate, n_rounds):
        n_evaluations = 0
        score_evaluations = []
        for i in range(n_rounds):
            accept_function = get_accept_function(i, temperature, cooling_rate)
            # population = parallel([delayed(mutate)(m, mapspace_globals, accept_function) for m in population])
            for j, mapping in enumerate(population):
                population[j], evaluations = mutate(mapping, mapspace_globals, accept_function)
                n_evaluations += evaluations
            best_score = min(m.prev_score for m in population)
            porp_in_10pct = sum(m.prev_score < best_score * 1.1 for m in population) / len(population)
            t = time.time() - t0
            if i % 100 == 0:
                print(f"Thread {thread} iteration {i}/{n_rounds} ({t:.2f}s) ({n_evaluations} evaluations): {best_score:.2e}, {porp_in_10pct * 100:.2f}% within 10%")
            score_evaluations.append((best_score, n_evaluations))
            # best_mapping = min(population, key=lambda m: m.prev_score)
            # population = [copy.deepcopy(best_mapping) for _ in range(population_size_per_thread)]
        return population, score_evaluations

    population_size = 1000
    num_threads = 16
    n_rounds = 10000
    population = [Mapping(sims) for _ in range(ceil(population_size / num_threads))]
    results = parallel([delayed(anneal_population)(i, population, mapspace_globals, 0.07, 8, n_rounds) for i in range(num_threads)])
    pops, score_evaluations = zip(*results)
    aggregate_score = []
    aggregate_evaluations = []
    for se in score_evaluations:
        if not aggregate_score:
            aggregate_score = [s for s, _ in se]
            aggregate_evaluations = [e for _, e in se]
        else:
            for i, (s, e) in enumerate(se):
                aggregate_score[i] = min(aggregate_score[i], s)
                aggregate_evaluations[i] += e
                
    for score, evaluations in zip(aggregate_score, aggregate_evaluations):
        print(f"{evaluations}, {score}")
    
    
    mappings = list(itertools.chain(*pops))
    mappings = pd.concat([m.evaluate(mapspace_globals, return_df=True)[0] for m in mappings])
    return mappings


def fuse_sims_ga_mcts(
    sims: dict[str, list[SIM]],
    mapspace_globals: MapsapceGlobals,
):
    t0 = time.time()
    
    def mutate(mapping: Mapping, mapspace_globals: MapsapceGlobals, accept_function: callable):
        prev_mapping = copy.deepcopy(mapping)
        prev_score = mapping.prev_score
        try:
            choice = random.choice(mapping.get_mutation_functions())
            choice(mapspace_globals)
        except FailedMutation:
            return prev_mapping
        prev_score = mapping.prev_score
        new_score = mapping.evaluate(mapspace_globals)
        if new_score == float("inf"):
            return prev_mapping
        if accept_function(prev_score, new_score):
            return mapping
        return prev_mapping
    
    def get_accept_function(iteration, temperature, cooling_rate):
        proportion = iteration / n_rounds
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
            if random.random() < exp((prev_score - new_score) / prev_score / new_temp):
                return True
            return False
        return accept
    
    t0 = time.time()
    def anneal_population(population, mapspace_globals: MapsapceGlobals, temperature, cooling_rate, n_rounds):
        for i in range(n_rounds):
            accept_function = get_accept_function(i, temperature, cooling_rate)
            # population = parallel([delayed(mutate)(m, mapspace_globals, accept_function) for m in population])
            for j, mapping in enumerate(population):
                mapping = mutate(mapping, mapspace_globals, accept_function)
                population[j] = mapping.mcts_loops_intra(mapspace_globals)
            best_score = min(m.prev_score for m in population)
            porp_in_10pct = sum(m.prev_score < best_score * 1.1 for m in population) / len(population)
            t = time.time() - t0
            if i % 100 == 0:
                print(f"Iteration {i}/{n_rounds} ({t:.2f}s): {best_score:.2e}, {porp_in_10pct * 100:.2f}% within 10%")
            # best_mapping = min(population, key=lambda m: m.prev_score)
            # population = [copy.deepcopy(best_mapping) for _ in range(population_size_per_thread)]
        return population

    population_size_per_thread = 1
    num_threads = 8
    n_rounds = 1000
    population = [Mapping(sims) for _ in range(population_size_per_thread)]
    pops = parallel([delayed(anneal_population)(population, mapspace_globals, 0.07, 8, n_rounds) for _ in range(num_threads)])
    mappings = list(itertools.chain(*pops))
    mappings = pd.concat([m.evaluate(mapspace_globals, return_df=True) for m in mappings])
    return mappings

def fuse_sims(
    sims: dict[str, list[SIM]],
    pairwise_equivalent_ranks: PairwiseEquivalentRanks,
    einsum2ranks: dict[str, set[str]],
    resource2capacity: dict = None,
    return_nmappings_nbuckets: bool = False,
    lookahead_filter: bool = False,
):

    objective_function_cols = None
    cols = next(iter(sims.values()))[0].mapping.data.columns
    if objective_function_cols is None:
        objective_function_cols = [c for c in cols if not is_special_col(c)]
    keepcols = [MAPPING]
    if VALID in cols:
        keepcols.append(VALID)

    for sim_list in sims.values():
        for sim in sim_list:
            for col in objective_function_cols:
                if col not in sim.mapping.data.columns:
                    sim.mapping.data[col] = 0
            sim.mapping.data = sim.mapping.data[objective_function_cols + [MAPPING, VALID]]

    mapspace_globals = MapsapceGlobals(
        sims,
        einsum2ranks,
        pairwise_equivalent_ranks,
        resource2capacity,
        objective_function_cols,
    )
    
    return fuse_sims_simulated_anneal(
        sims,
        mapspace_globals
    )