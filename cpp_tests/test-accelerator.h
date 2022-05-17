#pragma once

#include <boost/test/unit_test.hpp>

// Timeloop
#include <mapping/parser.hpp>
#include <model/sparse-optimization-parser.hpp>
#include <workload/workload.hpp>

#include "pytimeloop/model/accelerator.h"

using namespace boost::unit_test;

// Two tests with different workloads cannot pass at the same time.
// Issue with Timeloop's global workload setup.
// BOOST_AUTO_TEST_CASE(test_accelerator_1level) {
//   using namespace pytimeloop::pymodel;

//   auto config = TestConfig::config_gen->GetModelOneLevelConfig();
//   auto root_node = config.getRoot();

//   auto arch_specs =
//       model::Engine::ParseSpecs(root_node.lookup("architecture"), false);

//   problem::Workload workload;
//   problem::ParseWorkload(root_node.lookup("problem"), workload);

//   auto mapping = mapping::ParseAndConstruct(root_node.lookup("mapping"),
//                                             arch_specs, workload);

//   auto sparse_opts =
//       sparse::ParseAndConstruct(config::CompoundConfigNode(), arch_specs);

//   Accelerator acc(arch_specs);
//   auto result = acc.Evaluate(mapping, workload, sparse_opts);

//   BOOST_TEST_REQUIRE(result.cycles == 48);
//   BOOST_TEST_REQUIRE(result.utilization == 1.00);
// };

BOOST_AUTO_TEST_CASE(test_accelerator_3level) {
  using namespace pytimeloop::pymodel;

  auto config = TestConfig::config_gen->GetModelThreeLevelSpatialConfig();
  auto root_node = config.getRoot();

  auto arch_specs =
      model::Engine::ParseSpecs(root_node.lookup("architecture"), false);

  problem::Workload workload;
  problem::ParseWorkload(root_node.lookup("problem"), workload);

  auto mapping = mapping::ParseAndConstruct(root_node.lookup("mapping"),
                                            arch_specs, workload);

  auto sparse_opts =
      sparse::ParseAndConstruct(config::CompoundConfigNode(), arch_specs);

  Accelerator acc(arch_specs);
  auto result = acc.Evaluate(mapping, workload, sparse_opts);

  BOOST_TEST_REQUIRE(result.cycles == 3072);
  BOOST_TEST_REQUIRE(result.utilization == 1.00);
};
