from collections.abc import Mapping
from numbers import Real

from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
from pytimeloop.timeloopfe.v4.ert import Ert
from pytimeloop.looptree.accesses import *
from pytimeloop.looptree.mapping_utilities import *


def gather_actions(looptree_results, mapping, workload, bindings):
    reads, writes = reads_and_writes_from_fill_by_parent(
        looptree_results.fills_by_parent,
        mapping,
        workload
    )
    reads, writes = get_total_accesses(reads), get_total_accesses(writes)

    peer_reads, peer_writes = reads_and_writes_from_fill_by_peer(
        looptree_results.fills_by_peer,
        mapping,
        workload
    )
    peer_reads = get_total_accesses(peer_reads)
    peer_writes = get_total_accesses(peer_writes)

    for k, v in peer_reads.items():
        if k in reads:
            reads[k] += v
        else:
            reads[k] = v

    for k, v in peer_writes.items():
        if k in writes:
            writes[k] += v
        else:
            writes[k] = v

    einsums_with_complete_mapping = get_einsums_with_complete_mappings(mapping['nodes'])
    einsum_id_to_name = workload.einsum_id_to_name()

    ops = sum(
        get_sum_of_pw_qpolynomial(v)
        for einsum_id, (tags, v) in looptree_results.ops.items()
        if einsum_id_to_name[einsum_id] in einsums_with_complete_mapping
    ).to_python()

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
        energy_per_ac = ert.find_component(component).find_action(action).energy
        energy_result[(component, action)] = counts*energy_per_ac

    return energy_result

