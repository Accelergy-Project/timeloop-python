import unittest
from pathlib import Path

from bindings.config import Config
from bindings.looptree import LooptreeWorkload

from tests.util import TEST_TMP_DIR, gather_yaml_configs


class LooptreeWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        yaml_str = gather_yaml_configs(Path(__file__).parent / 'test_configs',
                                       ['looptree-test.yaml'])
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

    def assert_maps_are_inverted_equivalent(self, dict1, dict2):
        for key1, value1 in dict1.items():
            self.assertEqual(key1, dict2[value1])

        for key2, value2 in dict2.items():
            self.assertEqual(key2, dict1[value2])

        self.assertEqual(set(dict1.keys()), set(dict2.values()))
        self.assertEqual(set(dict1.values()), set(dict2.keys()))