import unittest

from bindings.looptree import *

from tests.load_config_mixin import LoadConfigMixin


class TestAnalyzer(LoadConfigMixin, unittest.TestCase):
    def test_matmul(self):
        config, spec = self.load_config([
            'cascaded_mm_multi_32.workload.yaml',
            'four_level.arch.yaml'
        ])

        workload = LooptreeWorkload.parse_cfg(config.root['problem'])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
        REF = {
            'M1': {'C2'},
            'C1': set(),
            'P1': {'P2'},
            'P2': {'P1', 'P3'}
        }
        rank_name_to_id = workload.dimension_name_to_id()
        rank_id_to_name = workload.dimension_id_to_name()
        for rank_name, ref_equivalent_rank_names in REF.items():
            rank_id = rank_name_to_id[rank_name]
            equivalent_rank_ids = analyzer.pairwise_equivalent_dimensions(rank_id)
            equivalent_rank_names = {rank_id_to_name[i] for i in equivalent_rank_ids}
            self.assertEqual(equivalent_rank_names, ref_equivalent_rank_names)

    def test_mha(self):
        config, spec = self.load_config([
            'cascaded_mha.workload.yaml',
            'four_level.arch.yaml'
        ])

        workload = LooptreeWorkload.parse_cfg(config.root['problem'])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
        REF = {
            'BK': {'BQ', 'BV', 'BQK'},
            'MQ': {'MK', 'MV', 'MQK'},
            'MK': {'MQ', 'MV', 'PQK'},
            'MV': {'MQ', 'MK', 'PAV'}
        }
        rank_name_to_id = workload.dimension_name_to_id()
        rank_id_to_name = workload.dimension_id_to_name()
        for rank_name, ref_equivalent_rank_names in REF.items():
            rank_id = rank_name_to_id[rank_name]
            equivalent_rank_ids = analyzer.pairwise_equivalent_dimensions(rank_id)
            equivalent_rank_names = {rank_id_to_name[i] for i in equivalent_rank_ids}
            self.assertEqual(equivalent_rank_names, ref_equivalent_rank_names)