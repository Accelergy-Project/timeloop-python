from collections import defaultdict
from collections.abc import Callable, Set
from itertools import combinations, product

from .linear_mapping import LinearMapping
from pytimeloop.looptree.mapping_utilities import get_last_storage_node

def make_storage(
    mapping: LinearMapping,
    level,
    must_retain_tensors: Set,
    can_retain_tensors: Set,
    tensor_to_relevant_ranks,
    explore_uneven,
    must_fully_reuse_tensors: Set=None,
    add_split_at_tensors: Set=None,
    must_have_terminal_storage: bool=False,
    logfunc: Callable=None,
    return_retained_tensors: bool=False,
    automatically_lower_below_relevant_ranks: bool = False
):
    if logfunc is None:
        logfunc = lambda msg: None  # do nothing

    if add_split_at_tensors is None:
        add_split_at_tensors = set()

    tensors = must_retain_tensors | can_retain_tensors

    if must_fully_reuse_tensors is None:
        must_fully_reuse_tensors = set()

    # Further mutated mappings copy from original first.
    original = mapping

    if not explore_uneven:
        for r in range(len(can_retain_tensors)+1):
            for also_retained_tensors in combinations(can_retain_tensors, r):
                mapping = original.copy()

                retained_tensors = must_retain_tensors | set(also_retained_tensors)
                mapping.add_storage(level, retained_tensors)
                if any(t in add_split_at_tensors for t in retained_tensors):
                    mapping.add_sequential()

                if return_retained_tensors:
                    yield mapping, retained_tensors
                else:
                    yield mapping
        return

    tensors = list(sorted(tensors))

    all_tensor_choices = []
    for tensor_id in tensors:
        tensor_must_be_fully_reused = tensor_id in must_fully_reuse_tensors

        relevant_ranks = tensor_to_relevant_ranks[tensor_id]
        tensor_choices = []
        last_is_relevant = True

        min_i = get_last_storage_node(mapping, tensor_id)
        for i, node in enumerate(mapping[min_i+1:]):
            i += min_i+1
            if node["type"] == "temporal":
                rank_id = node["rank"]
                is_relevant = rank_id in relevant_ranks
                if ((last_is_relevant and not is_relevant)
                    or not automatically_lower_below_relevant_ranks):
                    # Choice 1: fused
                    tensor_choices.append(i)
                    if tensor_must_be_fully_reused:
                        break
                last_is_relevant = is_relevant

        # There has not been a single irrelevant loop
        if last_is_relevant and (not tensor_must_be_fully_reused
                                 or len(tensor_choices) == 1):
            tensor_choices.append(len(mapping))

        if tensor_id in can_retain_tensors:
            tensor_choices.append(None)

        all_tensor_choices.append(tensor_choices)

    for choices in product(*all_tensor_choices):
        if must_have_terminal_storage:
            if not any(c == len(original) for c in choices):
                continue

        # Collect tensors with the same idx
        retained_tensors = set()
        idx_to_tensors = defaultdict(list)
        for idx, tensor in zip(choices, tensors):
            if idx is not None:
                idx_to_tensors[idx].append(tensor)
                retained_tensors.add(tensor)

        mapping = original.copy()
        success = True
        for idx, tensors_at_idx in sorted(idx_to_tensors.items(),
                                          key=lambda pair: pair[0],
                                          reverse=True):
            if any(t in add_split_at_tensors for t in tensors_at_idx):
                mapping.add_sequential(idx)
            mapping.add_storage(level, tensors_at_idx, idx)
            for t in tensors_at_idx:
                for node in mapping[:idx]:
                    if node["type"] == "storage" and t in node["dspace"]:
                        break
                    if node["type"] == "temporal" and node["rank"] not in tensor_to_relevant_ranks[t]:
                        success = False
                        break
        if not success:
            continue

        assert retained_tensors & must_retain_tensors == must_retain_tensors

        if return_retained_tensors:
            yield mapping, retained_tensors
        else:
            yield mapping