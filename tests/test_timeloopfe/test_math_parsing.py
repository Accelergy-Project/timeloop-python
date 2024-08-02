import os
import unittest
from pytimeloop.timeloopfe.v4.specification import Specification
from pytimeloop.timeloopfe.v4.arch import (
    Component,
    Hierarchical,
    Parallel,
    Pipelined,
)
from pytimeloop.timeloopfe.v4.processors import References2CopiesProcessor


class TestMathProcessorParsing(unittest.TestCase):
    def get_spec(self, **kwargs) -> Specification:
        this_script_dir = os.path.dirname(os.path.realpath(__file__))
        return Specification.from_yaml_files(
            os.path.join(this_script_dir, "arch_nest.yaml"), **kwargs
        )

    def test_math_parsing(self):
        spec = self.get_spec(processors=[References2CopiesProcessor])
        arch = spec.architecture.nodes
        arch[0].attributes["test"] = "1 + 1"
        arch[0].attributes["test2"] = "1 + known_value"
        arch[0].attributes["test3"] = "len('abcd')"
        spec.variables["known_value"] = 2
        spec.parse_expressions()
        arch = spec.architecture.nodes
        self.assertEqual(arch[0].attributes["test"], 2)
        self.assertEqual(arch[0].attributes["test2"], 3)
        self.assertEqual(arch[0].attributes["test3"], 4)

    def test_math_parsing_fail(self):
        spec = self.get_spec(processors=[References2CopiesProcessor])
        arch = spec.architecture.nodes
        arch[0].attributes["test"] = "intentionally invalid math. should fail."
        with self.assertRaises(ArithmeticError):
            spec.parse_expressions()
