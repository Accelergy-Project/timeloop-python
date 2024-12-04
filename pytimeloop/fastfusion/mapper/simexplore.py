from collections.abc import Mapping
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pytimeloop.fastfusion.sim import SIM
from pytimeloop.fastfusion.pareto import Pareto

def explore_fusion(einsum_to_result: Mapping):

    r2 = {}
    for einsum_id, compat_dict in einsum_to_result.items():
        r2[einsum_id] = Parallel(n_jobs=1)(delayed(paretofy)(k, v) for k, v in compat_dict.items())

    # for einsum_id, compat_dict in result.items():
    #     r2[einsum_id] = [SIM(k, Pareto(pd.DataFrame(v).fillna(0))) for k, v in compat_dict.items()]
        
    sims = list(r2.values())
    s = sims.pop(0)


    while sims:
        live_tensors = set.union(set(), *[sim[0].tensor_names for sim in sims])
        ns = sims.pop(0)
        next_live_tensors = set.union(set(), *[sim[0].tensor_names for sim in sims])

        for s2 in s:
            s2.consolidate(live_tensors)

        ns = SIM.combine_combineable(ns, next_live_tensors | s[0].tensor_names)
        ns = SIM.group_by_left(ns, s[0].tensor_names)
        s = SIM.combine_combineable(s, live_tensors)
        s = SIM.group_by_right(s, live_tensors)

        print("\n\n")
        print("\n\n" + "=" * 100 + f"\n{len(sims) + 1} Remaining\n" + "=" * 100)

        DO_PRINT = False

        with open('s_keys.txt', 'w') as f:
            for key in sorted(s.keys()):
                f.write(f"{key}\n")

        with open('s2_keys.txt', 'w') as f:
            for key in sorted(ns.keys()):
                f.write(f"{key}\n")

        combined: list[SIM] = []
        for k in s:
            if k in ns:
                for a, b in itertools.product(s[k], ns[k]):
                    if DO_PRINT:
                        print(f"\t{a.tiling_str()} <--> {b.tiling_str()}")
                    combined.append(a.merge_next(b, set(), delay=True))
                    # combined_keys.append()
            elif DO_PRINT:
                print(f"\tNo match for {s[k][0].tiling_str()}")

        for c, mapping in zip(combined, Parallel(n_jobs=128)(c.mapping for c in combined)):
            c.mapping = mapping

        s = combined
        print(f"Generated {len(s)} solutions")
        
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