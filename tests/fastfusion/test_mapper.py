import itertools
import sys
import unittest

from bindings.looptree import LooptreeWorkload

from pytimeloop.fastfusion.mapper.mapper2 import mapper, MacArrayConstraint

from tests.load_config_mixin import LoadConfigMixin
from tests.util import TEST_TMP_DIR


class TestMapper(LoadConfigMixin, unittest.TestCase):
    def test_mapper(self):
        config, spec = self.load_config([
            'cascaded_mm_large.workload.yaml',
            'four_level.arch.yaml'
        ])

        mac_constraint = MacArrayConstraint(
            64,
            64,
            {
                'Fc1': 'Filter1',
                'Fc2': 'Filter2'
            },
            {
                'Fc1': 'M1',
                'Fc2': 'M2'
            },
            {
                'Fc1': 'C1',
                'Fc2': 'C2'
            }
        )

        result = mapper(config,
                        mac_constraint,
                        spec,
                        tmp_path=TEST_TMP_DIR,
                        verbose_stream=sys.stdout)
        
        import pandas as pd
        from pytimeloop.fastfusion.sim import SIM
        from pytimeloop.fastfusion.pareto import Pareto
        r2 = {}
        for einsum_id, compat_dict in result.items():
            r2[einsum_id] = [SIM(k, Pareto(pd.DataFrame(v))) for k, v in compat_dict.items()]
            
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
                        combined.append(a.copy())
                        combined[-1].merge_next(b, set())
                        # combined_keys.append()
                elif DO_PRINT:
                    print(f"\tNo match for {s[k][0].tiling_str()}")

            s = combined
            print(f"Generated {len(s)} solutions")
            
        print(s)


    # def test_fusion():
    #     config, spec = self.load_config([
    #         'cascaded_mm_small.workload.yaml',
    #         'four_level.arch.yaml'
    #     ])
    #     mac_constraint = MacArrayConstraint(
    #         64,
    #         64,
    #         {
    #             'Fc1': 'Filter1',
    #             'Fc2': 'Filter2'
    #         },
    #         {
    #             'Fc1': 'M1',
    #             'Fc2': 'M2'
    #         },
    #         {
    #             'Fc1': 'C1',
    #             'Fc2': 'C2'
    #         }
    #     )
    #     result = mapper(config,
    #                     mac_constraint,
    #                     spec,
    #                     tmp_path=TEST_TMP_DIR,
    #                     verbose_stream=sys.stdout)
    #     print(result)


if __name__ == '__main__':
    unittest.main(failfast=True)
    