import unittest

from pytimeloop.fastfusion.mapper.per_einsum_subspaces.subspaces.tile_shape.shape_subspace import ShapeSubspace

from tests.load_config_mixin import LoadConfigMixin


class TestShapeSubspace(LoadConfigMixin, unittest.TestCase):
    def setUp(self) -> None:
        einsum_shape = {
            0: 4,
            1: 2,
            2: 2
        }
        shape_subspace = ShapeSubspace(einsum_shape, [0, 1, 0, 2, 2])
        self.subspace_it = iter(shape_subspace)

    def test_iterate_all(self):
        self.assertEqual(36, self.count_iterations(self.subspace_it))

    def test_skip_first_iteration(self):
        # Simulates first choice not being valid
        first_choice = next(self.subspace_it)
        self.subspace_it.skip_current_rank_iteration()
        self.assertEqual(0, self.count_iterations(self.subspace_it))

    def test_skip_second_rank_interation(self):
        first_choice = next(self.subspace_it)
        second_rank_val_in_first_choice = first_choice[2]
        for shape in self.subspace_it:
            if shape[2] != second_rank_val_in_first_choice:
                break
        self.subspace_it.skip_current_rank_iteration()
        self.assertEqual(24, self.count_iterations(self.subspace_it))

    @staticmethod
    def count_iterations(it):
        count = 0
        for p in it:
            count += 1
        return count