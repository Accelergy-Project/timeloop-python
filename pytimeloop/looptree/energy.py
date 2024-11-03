from collections.abc import Mapping
from numbers import Real

from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
from pytimeloop.timeloopfe.v4.ert import Ert
from pytimeloop.looptree.accesses import *
from pytimeloop.looptree.mapping_utilities import *

def gather_actions(looptree_results, mapping, workload, bindings, is_path=False):
    reads, writes = reads_and_writes_from_fill_by_parent(
        looptree_results.fills_by_parent,
        mapping,
        workload,
        is_path
    )
    reads, writes = get_total_accesses(reads), get_total_accesses(writes)

    peer_reads, peer_writes = reads_and_writes_from_fill_by_peer(
        looptree_results.fills_by_peer,
        mapping,
        workload,
        is_path
    )
    peer_reads = get_total_accesses(peer_reads)
    peer_writes = get_total_accesses(peer_writes)

    for k, v in peer_reads.items():
        reads[k] = reads.get(k, 0) + v

    for k, v in peer_writes.items():
        writes[k] = writes.get(k, 0) + v

    einsum_name_to_id = workload.einsum_name_to_id()

    einsums_with_complete_mapping = get_einsums_with_complete_mappings(mapping['nodes'], workload, is_path)
    einsums_with_complete_mapping = {
        e if isinstance(e, int) else einsum_name_to_id[e]
        for e in einsums_with_complete_mapping
    }

    ops = gather_ops(looptree_results.ops, einsums_with_complete_mapping)

    actions = {}
    for (buf, tensor, einsum), counts in reads.items():
        buf = bindings[buf]
        key = (buf, 'read')
        if key not in actions:
            actions[key] = 0
        actions[key] += counts

    for (buf, tensor, einsum), counts in writes.items():
        buf = bindings[buf]
        key = (buf, 'write')
        if key not in actions:
            actions[key] = 0
        actions[key] += counts

    actions[(bindings[max(bindings.keys())], 'compute')] = ops

    return actions


def compute_energy_from_actions(action_counts: Mapping[(str, str), Real],
                   ert: Ert):
    energy_result = {}
    for (component, action), counts in action_counts.items():
        if counts == 0:
            continue
        component_table = ert.find_component(component)
        action_table = component_table.find_action(action)
        if action_table is None:
            raise RuntimeError(
                f'Could not find action {action} for component {component}'
            )
        energy_per_ac = action_table.energy
        energy_result[(component, action)] = counts*energy_per_ac

    return energy_result


def gather_ops(ops, einsums_with_complete_mapping):
    total = 0
    for einsum_id, (tags, v) in ops.items():
        if einsum_id not in einsums_with_complete_mapping:
            continue
        if isinstance(v, isl.PwQPolynomial):
            total += get_sum_of_pw_qpolynomial(v).to_python()
        elif isinstance(v, Number):
            total += v
        else:
            total += v
    return total