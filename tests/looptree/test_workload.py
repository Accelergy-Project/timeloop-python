import unittest
from pathlib import Path
import pickle

from bindings.config import Config
from bindings.looptree import *

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class TestLooptreeWorkload(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        yaml_str = gather_yaml_configs(
            Path(__file__).parent.parent / 'test_configs',
            ['cascaded_mm.workload.yaml']
        )
        config = Config(yaml_str, 'yaml')
        cls._workload = LooptreeWorkload.parse_cfg(config.root['problem'])

    def test_einsum_name_to_id(self):
        name_to_id = self._workload.einsum_name_to_id()
        id_to_name = self._workload.einsum_id_to_name()
        self.assert_maps_are_inverted_equivalent(name_to_id, id_to_name)

    def test_dspace_name_to_id(self):
        name_to_id = self._workload.data_space_name_to_id()
        id_to_name = self._workload.data_space_id_to_name()
        self.assert_maps_are_inverted_equivalent(name_to_id, id_to_name)

    def test_rank_shape(self):
        name_to_id = self._workload.dimension_name_to_id()
        rank_shape = self._workload.get_rank_shape(name_to_id['P1'])
        self.assertEqual((0, 8), rank_shape)
        rank_shape = self._workload.get_rank_shape(name_to_id['M2'])
        self.assertEqual((0, 7), rank_shape)

    def test_tensor_volume(self):
        name_to_id = self._workload.data_space_name_to_id()
        self.assertEqual(18, self._workload.get_tensor_volume(name_to_id['Fmap1']))
        self.assertEqual(8, self._workload.get_tensor_volume(name_to_id['Filter1']))
        self.assertEqual(36, self._workload.get_tensor_volume(name_to_id['Fmap2']))
        self.assertEqual(32, self._workload.get_tensor_volume(name_to_id['Filter2']))
        self.assertEqual(72, self._workload.get_tensor_volume(name_to_id['Fmap3']))

    def test_pickling(self):
        pickled_workload = pickle.dumps(self._workload)
        unpickled_workload = pickle.loads(pickled_workload)

        self.assertEqual(unpickled_workload.einsum_name_to_id(),
                         self._workload.einsum_name_to_id())
        self.assertEqual(unpickled_workload.data_space_name_to_id(),
                         self._workload.data_space_name_to_id())
        self.assertEqual(unpickled_workload.dimension_name_to_id(),
                         self._workload.dimension_name_to_id())

        dspace_name_to_id = unpickled_workload.data_space_name_to_id()
        for _, dspace_id in dspace_name_to_id.items():
            self.assertEqual(
                unpickled_workload.get_tensor_volume(dspace_id),
                self._workload.get_tensor_volume(dspace_id),
            )


    def assert_maps_are_inverted_equivalent(self, dict1, dict2):
        for key1, value1 in dict1.items():
            self.assertEqual(key1, dict2[value1])

        for key2, value2 in dict2.items():
            self.assertEqual(key2, dict1[value2])

        self.assertEqual(set(dict1.keys()), set(dict2.values()))
        self.assertEqual(set(dict1.values()), set(dict2.keys()))


class TestLooptreeWorkloadDependencyAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        yaml_str = gather_yaml_configs(
            Path(__file__).parent.parent / 'test_configs',
            ['cascaded_mm.workload.yaml']
        )
        config = Config(yaml_str, 'yaml')
        cls._workload = LooptreeWorkload.parse_cfg(config.root['problem'])
        cls._analyzer = LooptreeWorkloadDependencyAnalyzer(cls._workload)
        cls._einsum_name_to_id = cls._workload.einsum_name_to_id()
        cls._data_space_name_to_id = cls._workload.data_space_name_to_id()
        cls._dimension_name_to_id = cls._workload.dimension_name_to_id()

    def test_find_dependency_chain(self):
        self.assertEqual(
            [[0, 1]],
            self._analyzer.find_einsum_dependency_chain(0, 1)
        )

        self.assertEqual(
            [[0]],
            self._analyzer.find_einsum_dependency_chain(0, 0)
        )

    def test_einsum_dim_is_directly_relevant(self):
        ARG_TO_ANSWER = {
            ('Fc1', 'P1', 'Fmap1'): True,
            ('Fc1', 'P1', 'Filter1'): False,
            ('Fc1', 'C1', 'Filter1'): True,
            ('Fc2', 'P2', 'Fmap2'): True
        }
        for (einsum, rank, tensor), answer in ARG_TO_ANSWER.items():
            self.assertEqual(
                answer,
                self._analyzer.einsum_dim_is_directly_relevant_to_tensor(
                    self._einsum_name_to_id[einsum],
                    self._dimension_name_to_id[rank],
                    self._data_space_name_to_id[tensor]
                )
            )

    def test_einsum_dim_is_relevant(self):
        ARG_TO_ANSWER = {
            ('Fc1', 'P1', 'Fmap1'): True,
            ('Fc1', 'P1', 'Filter1'): False,
            ('Fc1', 'C1', 'Filter1'): True,
            ('Fc2', 'P2', 'Fmap1'): True,
            ('Fc2', 'P2', 'Fmap2'): True
        }
        for (einsum, rank, tensor), answer in ARG_TO_ANSWER.items():
            # print(einsum, rank, tensor)
            self.assertEqual(
                answer,
                self._analyzer.einsum_dim_is_relevant_to_tensor(
                    self._einsum_name_to_id[einsum],
                    self._dimension_name_to_id[rank],
                    self._data_space_name_to_id[tensor]
                )
            )

    def test_einsum_dims_directly_relevant(self):
        ARG_TO_ANSWER = {
            ('Fc1', 'Fmap1'): {'P1', 'C1'},
            ('Fc1', 'Filter1'): {'M1', 'C1'},
            ('Fc2', 'Fmap2'): {'P2', 'C2'},
            ('Fc2', 'Filter1'): {},
            ('Fc2', 'Fmap1'): {}
        }

        for (einsum, tensor), answer in ARG_TO_ANSWER.items():
            answer = set(
                self._dimension_name_to_id[a] for a in answer
            )
            self.assertEqual(
                answer,
                self._analyzer.einsum_dims_directly_relevant_to_tensor(
                    self._einsum_name_to_id[einsum],
                    self._data_space_name_to_id[tensor]
                )
        )

    def test_einsum_dims_relevant(self):
        ARG_TO_ANSWER = {
            ('Fc1', 'Fmap1'): {'P1', 'C1'},
            ('Fc1', 'Filter1'): {'M1', 'C1'},
            ('Fc2', 'Fmap2'): {'P2', 'C2'},
            ('Fc2', 'Filter1'): {'C2'},
            ('Fc2', 'Fmap1'): {'P2'}
        }

        for (einsum, tensor), answer in ARG_TO_ANSWER.items():
            answer = set(
                self._dimension_name_to_id[a] for a in answer
            )
            self.assertEqual(
                answer,
                self._analyzer.einsum_dims_relevant_to_tensor(
                    self._einsum_name_to_id[einsum],
                    self._data_space_name_to_id[tensor]
                )
        )

    def test_equivalent_ranks(self):
        ARG_TO_ANSWER = {
            ('Fc1', 'P1'): {'P1', 'P2'},
            ('Fc1', 'C1'): {'C1'},
            ('Fc1', 'M1'): {'M1', 'C2'},
            ('Fc2', 'P2'): {'P1', 'P2'},
            ('Fc2', 'C2'): {'M1', 'C2'},
            ('Fc2', 'M2'): {'M2'},
        }
        for (einsum, einsum_rank), answer in ARG_TO_ANSWER.items():
            equiv_ranks = self._analyzer.equivalent_dimensions(
                self._einsum_name_to_id[einsum],
                self._dimension_name_to_id[einsum_rank]
            )
            answer_id = {self._dimension_name_to_id[r] for r in answer}
            self.assertTrue(answer_id, equiv_ranks)
