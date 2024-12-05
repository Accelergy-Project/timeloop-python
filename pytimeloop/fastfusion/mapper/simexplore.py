from collections.abc import Mapping
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pytimeloop.fastfusion.sim import SIM
from pytimeloop.fastfusion.pareto import Pareto

def explore_fusion(einsum_to_result: Mapping, resource2capacity: dict=None):
    resource2capacity = resource2capacity or {}

    r2 = {}
    for einsum_id, compat_dict in einsum_to_result.items():
        r2[einsum_id] = Parallel(n_jobs=1)(delayed(paretofy)(k, v) for k, v in compat_dict.items())

    # for einsum_id, compat_dict in result.items():
    #     r2[einsum_id] = [SIM(k, Pareto(pd.DataFrame(v).fillna(0))) for k, v in compat_dict.items()]
        
    sims = list(r2.values())
    s = sims.pop(0)


    while sims:
        print("\n\n")
        print("\n\n" + "=" * 100 + f"\n{len(sims) + 1} Remaining\n" + "=" * 100)
        live_tensors = set.union(set(), *[sim[0].tensor_names for sim in sims])
        ns = sims.pop(0)
        next_live_tensors = set.union(set(), *[sim[0].tensor_names for sim in sims])

        for s2 in s:
            s2.consolidate(live_tensors, resource2capacity)
        next_next_live_tensors = next_live_tensors | s[0].tensor_names
        ns = SIM.combine_combineable(ns, next_next_live_tensors)
        ns = SIM.group_by_left(ns, s[0].tensor_names)
        # for s2 in ns:
        #     s2.consolidate(live_tensors, resource2capacity)
        print(f"\tNEXT: Combined by {sorted(next_next_live_tensors)}")
        print(f"\tNEXT: Grouped by {sorted(s[0].tensor_names)}")
        print(f"\tPREV: Combined by {sorted(live_tensors)}")
        print(f"\tPREV: Grouped by {sorted(live_tensors)}")
        s = SIM.combine_combineable(s, live_tensors)
        s = SIM.group_by_right(s, live_tensors)

        DO_PRINT = True

        combined: list[SIM] = []
        for k in s:
            if k in ns:
                for a, b in itertools.product(s[k], ns[k]):
                    if DO_PRINT:
                        print(f"\t{a.tiling_str()} {a.get_shared_loop_index(live_tensors)} <--> {b.tiling_str()}{b.get_shared_loop_index(next_next_live_tensors)}. ({len(a.mapping.data)})x({len(b.mapping.data)})")
                    if len(a.mapping.data) * len(b.mapping.data) > 1e4:
                        print(f'Many mappings detected: {len(a.mapping.data) * len(b.mapping.data)}')
                        # a.merge_next(b, set(), next_live_tensors, resource2capacity, delay=False)

                    combined.append(a.merge_next(b, set(), next_live_tensors, resource2capacity, delay=True))
            elif DO_PRINT:
                print(f"\tNo match for {k} ||||||||| {s[k][0].tiling_str()}")

        for c, mapping in zip(combined, Parallel(n_jobs=128)(c.mapping for c in combined)):
            c.mapping = mapping
        # for c, mapping in zip(combined, [c.mapping for c in combined]):
            # c.mapping = mapping[0](*mapping[1], **mapping[2])

        print(f"\tCombining {sum(len(s2) for s2 in s)}({len(s)}) x {sum(len(s2) for s2 in ns)}({len(ns)}) -> {len(combined)}")
        if DO_PRINT:
            for k in ns:
                if k not in s:
                    print(f"\tREVERSE: No match for {k} ||||||||| {ns[k][0].tiling_str()}")
        s = combined
        print(f'Number of buckets: {len(s)}')
        print(f'Number of mappings: {sum(len(s2.mapping.data) for s2 in s)}')
        print(f'Mappings per bucket: {sum(len(s2.mapping.data) for s2 in s) / len(s)}')
        
    for s2 in s:
        s2.consolidate(set())
    s_final = SIM.combine_combineable(s, set())[0]
    data = s_final.mapping.data
    # Sort data by the columns "Latency" and "Energy"
    last_level_occupancy = None
    for i in reversed(range(3)):
        if f"RESOURCE_1_LEVEL_{i}" not in data:
            continue
        if last_level_occupancy is not None:
            non_left_cur_level_occupancy = data[f"RESOURCE_1_LEVEL_{i}"] + last_level_occupancy
        else:
            non_left_cur_level_occupancy = data[f"RESOURCE_1_LEVEL_{i}"]
        left_cur_level_occupancy = data[f"RESOURCE_1_LEFT_LEVEL_{i}"]
        last_level_occupancy = np.maximum(non_left_cur_level_occupancy,
                                            left_cur_level_occupancy)
    data["Occupancy"] = last_level_occupancy

    return data


def paretofy(k, v):
    return SIM(k, Pareto(pd.DataFrame(v).fillna(0)))