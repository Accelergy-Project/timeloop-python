import sys
import unittest

from bindings.looptree import LooptreeWorkload

from pytimeloop.fastfusion.mapper.mapper import mapper
from pytimeloop.fastfusion.mapper.shape_subspace import ShapeSubspace

from tests.load_config_mixin import LoadConfigMixin
from tests.util import TEST_TMP_DIR


class TestMapper(LoadConfigMixin, unittest.TestCase):
    def test_mapper(self):
        config, spec = self.load_config([
            'cascaded_mm.workload.yaml',
            'three_level.arch.yaml'
        ])

        result = mapper(config,
                        spec,
                        TEST_TMP_DIR,
                        verbose_stream=sys.stdout)


class TestShapeSubspace(LoadConfigMixin, unittest.TestCase):
    def setUp(self) -> None:
        config, spec = self.load_config([
            'cascaded_mm.workload.yaml',
            'three_level.arch.yaml'
        ])

        workload = LooptreeWorkload.parse_cfg(config.root['problem'])
        NAME_OF_EINSUM_TO_MAP = 'Fc1'

        einsum_name_to_id = workload.einsum_name_to_id()
        id_of_einsum_to_eval = einsum_name_to_id[NAME_OF_EINSUM_TO_MAP]

        ranks = workload.einsum_ospace_dimensions(id_of_einsum_to_eval)
        einsum_shape = {
            rank_id: workload.get_rank_shape(rank_id)[1]+1 for rank_id in ranks
        }

        shape_subspace = ShapeSubspace(einsum_shape, ranks)
        self.subspace_it = iter(shape_subspace)

    def test_iterate_all(self):
        self.assertEqual(18, self.count_iterations(self.subspace_it))

    def test_skip_first_iteration(self):
        # Simulates first choice not being valid
        first_choice = next(self.subspace_it)
        self.subspace_it.skip_current_rank_iteration()
        self.assertEqual(0, self.count_iterations(self.subspace_it))

    def test_skip_second_rank_interation(self):
        first_choice = next(self.subspace_it)
        second_rank_val_in_first_choice = first_choice[0][-2]
        for shape in self.subspace_it:
            if shape[0][-2] != second_rank_val_in_first_choice:
                break
        self.subspace_it.skip_current_rank_iteration()
        self.assertEqual(12, self.count_iterations(self.subspace_it))

    @staticmethod
    def count_iterations(it):
        count = 0
        for _ in it:
            count += 1
        return count