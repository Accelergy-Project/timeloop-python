from collections.abc import Mapping
from numbers import Real

from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
from pytimeloop.timeloopfe.v4.ert import Ert
from pytimeloop.looptree.accesses import reads_and_writes_from_fill, reads_and_writes_from_ops, get_total_accesses


def gather_actions(looptree_results, mapping, workload, bindings):
    reads, writes = reads_and_writes_from_fill(looptree_results.fill,
                                               mapping,
                                               workload)
    ops_reads, ops_writes = reads_and_writes_from_ops(looptree_results.ops,
                                                      mapping,
                                                      workload)
    reads |= ops_reads
    writes |= ops_writes

    reads = get_total_accesses(reads)
    writes = get_total_accesses(writes)
    ops = sum(get_sum_of_pw_qpolynomial(v)
              for (tags, v) in looptree_results.ops.values())

    actions = {}
    for (buf, tensor), counts in reads.items():
        buf = bindings[buf]
        key = (buf, 'read')
        if key not in actions:
            actions[key] = 0
        actions[key] += counts

    for (buf, tensor), counts in writes.items():
        buf = bindings[buf]
        key = (buf, 'write')
        if key not in actions:
            actions[key] = 0
        actions[key] += counts

    actions[(bindings['compute'], 'compute')] = ops

    return actions


def compute_energy_from_actions(action_counts: Mapping[(str, str), Real],
                   ert: Ert):
    energy_result = {}
    for (component, action), counts in action_counts.items():
        energy_per_ac = ert.find_component(component).find_action(action).energy
        energy_result[(component, action)] = counts.to_python()*energy_per_ac

    return energy_result

