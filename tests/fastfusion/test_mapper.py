import itertools
import sys
import unittest
import logging

from bindings.looptree import LooptreeWorkload

from pytimeloop.fastfusion.mapper.mapper2 import mapper, PeArrayConstraint, MacArrayConstraint
from pytimeloop.fastfusion.mapper.mapper_snowcat import mapper as mapper_snowcat

from tests.load_config_mixin import LoadConfigMixin
from tests.util import TEST_TMP_DIR

from tests.load_config_mixin import CONFIG_DIR

class TestMapper(LoadConfigMixin, unittest.TestCase):
    def test_mapper(self):
        logging.basicConfig(filename='tests.fastfusion.test_mapper.log', level=logging.DEBUG)
        config, spec = self.load_config([
            'cascaded_mm_multi_32.workload.yaml',
            'four_level.arch.yaml'
        ])

        pe_constraint = PeArrayConstraint(4)

        mac_constraint = MacArrayConstraint(
            64,
            64,
            {f'Fc{x}': f'Filter{x}' for x in range(1, 32)},
            {f'Fc{x}': f'M{x}' for x in range(1, 32)},
            {f'Fc{x}': f'C{x}' for x in range(1, 32)}
        )

        result = mapper(config,
                        pe_constraint,
                        mac_constraint,
                        explore_glb_uneven=True,
                        explore_pe_uneven=False,
                        spec=spec,
                        tmp_path=TEST_TMP_DIR,
                        verbose_stream=sys.stdout)
        
        import pandas as pd
        from pytimeloop.fastfusion.sim import SIM
        from pytimeloop.fastfusion.pareto import Pareto
        r2 = {}
        
        def paretofy(k, v):
            return SIM(k, Pareto(pd.DataFrame(v).fillna(0)))
        
        from joblib import Parallel, delayed
        for einsum_id, compat_dict in result.items():
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

            DO_PRINT = True

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
                        combined.append(a.copy())
                        combined[-1].merge_next(b, set())
                        # combined_keys.append()
                elif DO_PRINT:
                    print(f"\tNo match for {s[k][0].tiling_str()}")

            s = combined
            print(f"Generated {len(s)} solutions")
            
        print(s)
        for s2 in s:
            s2.consolidate(set())
        s_final = SIM.combine_combineable(s, set())[0]
        data = s_final.mapping.data
        # Sort data by the columns "Latency" and "Energy"
        data = data.sort_values(by=["Latency", "Energy"])
        
        print(s_final)


class TestSnowcatMapper(LoadConfigMixin, unittest.TestCase):
    def test_mapper(self):
        logging.basicConfig(filename='tests.fastfusion.test_snowcat_mapper.log', level=logging.DEBUG)
        config, spec = self.load_config([
            'cascaded_mm_multi_32.workload.yaml',
            'snowcat.arch.yaml'
        ])

        result = mapper_snowcat(
            config,
            explore_glb_uneven=True,
            spec=spec,
            tmp_path=TEST_TMP_DIR,
            ffmt=True,
        )
        
        import pandas as pd
        from pytimeloop.fastfusion.sim import SIM
        from pytimeloop.fastfusion.pareto import Pareto
        r2 = {}
        
        def paretofy(k, v):
            return SIM(k, Pareto(pd.DataFrame(v).fillna(0)))
        
        from joblib import Parallel, delayed
        for einsum_id, compat_dict in result.items():
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

            DO_PRINT = True

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
                        combined.append(a.copy())
                        combined[-1].merge_next(b, set())
                        # combined_keys.append()
                elif DO_PRINT:
                    print(f"\tNo match for {s[k][0].tiling_str()}")

            s = combined
            print(f"Generated {len(s)} solutions")
            
        print(s)
        for s2 in s:
            s2.consolidate(set())
        s_final = SIM.combine_combineable(s, set())[0]
        data = s_final.mapping.data
        # Sort data by the columns "Latency" and "Energy"
        data = data.sort_values(by=["Latency", "Energy"])
        
        print(s_final)


if __name__ == '__main__':
    unittest.main(failfast=True)

