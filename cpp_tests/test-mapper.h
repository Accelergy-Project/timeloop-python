#pragma once

#include <boost/test/unit_test.hpp>

// Timeloop
#include <mapspaces/mapspace-factory.hpp>
#include <model/sparse-optimization-parser.hpp>
#include <search/search-factory.hpp>

#include "pytimeloop/mapper/coupled-mapper.h"

using namespace boost::unit_test;

BOOST_AUTO_TEST_CASE(test_coupled_mapper_three_level_spatial) {
  using namespace pytimeloop::pymapper;

  auto config = TestConfig::config_gen->GetMapperThreeLevelFreeBypassConfig();
  auto root_node = config.getRoot();

  problem::Workload workload;
  problem::ParseWorkload(root_node.lookup("problem"), workload);

  auto mapper_config = root_node.lookup("mapper");

  auto arch_config = root_node.lookup("architecture");
  bool is_sparse_topology = root_node.exists("sparse_optimizations");
  auto arch_specs = model::Engine::ParseSpecs(arch_config, is_sparse_topology);

  auto ert_config = root_node.lookup("ERT");
  arch_specs.topology.ParseAccelergyERT(ert_config);

  config::CompoundConfigNode sparse_opt_config;
  if (is_sparse_topology) {
    sparse_opt_config = root_node.lookup("sparse_optimizations");
  }
  auto sparse_opts = sparse::ParseAndConstruct(sparse_opt_config, arch_specs);
  workload.SetDefaultDenseTensorFlag(
      sparse_opts.compression_info.all_ranks_default_dense);

  int nthreads = std::thread::hardware_concurrency();
  mapper_config.lookupValue("num-threads", nthreads);

  std::vector<std::string> metrics;
  std::string metric;
  if (mapper_config.lookupValue("optimization-metric", metric)) {
    metrics = {metric};
  } else if (mapper_config.exists("optimization-metrics")) {
    mapper_config.lookupArrayValue("optimization-metrics", metrics);
  } else {
    metrics = {"edp"};
  }

  unsigned search_size = 0;
  mapper_config.lookupValue("search-size", search_size);
  if (search_size > 0) {
    search_size = 1 + (search_size - 1) / nthreads;
  }

  unsigned timeout = 1000;
  mapper_config.lookupValue("timeout", timeout);
  mapper_config.lookupValue("heartbeat", timeout);

  unsigned victory_condition = 500;
  mapper_config.lookupValue("victory-condition", victory_condition);

  auto penalize_consecutive_bypass_fails = false;
  mapper_config.lookupValue("penalize-consecutive-bypass-fails",
                            penalize_consecutive_bypass_fails);

  config::CompoundConfigNode arch_constr_config;
  if (arch_config.exists("constraints")) {
    arch_constr_config = arch_config.lookup("constraints");
  } else if (root_node.exists("arch_constraints")) {
    arch_constr_config = arch_config.lookup("arch_constraints");
  } else if (root_node.exists("architecture_constraints")) {
    arch_constr_config = arch_config.lookup("architecture_constraints");
  }

  config::CompoundConfigNode mapspace_config;
  if (root_node.exists("mapspace")) {
    mapspace_config = root_node.lookup("mapspace");
  } else if (root_node.exists("mapspace_constraints")) {
    mapspace_config = root_node.lookup("mapspace_constraints");
  }

  bool filter_spatial_fanout =
      sparse_opts.action_spatial_skipping_info.size() == 0;
  auto mapspace =
      mapspace::ParseAndConstruct(mapspace_config, arch_constr_config,
                                  arch_specs, workload, filter_spatial_fanout);
  auto split_mapspaces = mapspace->Split(nthreads);

  std::vector<search::SearchAlgorithm*> search_algs;
  auto search_config = root_node.lookup("mapper");
  for (int i = 0; i < nthreads; ++i) {
    search_algs.push_back(
        search::ParseAndConstruct(search_config, split_mapspaces.at(i), i));
  }

  std::vector<std::pair<mapspace::MapSpace*, search::SearchAlgorithm*>>
      mapspace_search_alg_pairs;
  for (int i = 0; i < nthreads; ++i) {
    mapspace_search_alg_pairs.emplace_back(split_mapspaces.at(i),
                                           search_algs.at(i));
  }

  CoupledMapper mapper(arch_specs, workload, mapspace_search_alg_pairs,
                       sparse_opts, metrics, search_size, timeout,
                       victory_condition, penalize_consecutive_bypass_fails);

  auto [best_mapping, best_result] = mapper.Run();

  BOOST_TEST_REQUIRE(best_result.energy == 45588.5,
                     boost::test_tools::tolerance(0.1));
  BOOST_TEST_REQUIRE(best_result.cycles == 1536);

  delete mapspace;
  for (auto s : search_algs) {
    delete s;
  }
}
