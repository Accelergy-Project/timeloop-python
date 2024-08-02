from tests.test_timeloopfe import (
    test_art,
    test_constraints,
    test_ert,
    test_nest,
    test_node,
    test_math_parsing,
    test_constraint_macro_parsing,
    test_load_examples,
    test_constraint_attach,
    test_compare_results,
    test_peer_dataspaces,
    test_refs2copies,
)
import unittest
import os
import shutil

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_art))
    suite.addTests(loader.loadTestsFromModule(test_constraints))
    suite.addTests(loader.loadTestsFromModule(test_ert))
    suite.addTests(loader.loadTestsFromModule(test_nest))
    suite.addTests(loader.loadTestsFromModule(test_node))
    suite.addTests(loader.loadTestsFromModule(test_math_parsing))
    suite.addTests(loader.loadTestsFromModule(test_constraint_macro_parsing))
    suite.addTests(loader.loadTestsFromModule(test_peer_dataspaces))
    suite.addTests(loader.loadTestsFromModule(test_load_examples))
    suite.addTests(loader.loadTestsFromModule(test_constraint_attach))
    suite.addTests(loader.loadTestsFromModule(test_compare_results))
    suite.addTests(loader.loadTestsFromModule(test_refs2copies))
    runner = unittest.TextTestRunner(verbosity=2, failfast=True)
    result = runner.run(suite)
    if result.wasSuccessful():
        shutil.rmtree(os.path.join(os.path.dirname(__file__),
                      "tests/test_timeloopfe/compare"))
