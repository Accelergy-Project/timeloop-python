#pragma once

#include <benchmark/benchmark.h>

// Timeloop
#include <compound-config/compound-config.hpp>
#include <mapspaces/mapspace-factory.hpp>
#include <model/sparse-optimization-parser.hpp>
#include <search/search-factory.hpp>
#include <workload/workload.hpp>

#include "pytimeloop/mapper/decoupled-mapper.h"

class ThreeLevelArchMapping : public benchmark::Fixture {
 public:
  std::unique_ptr<config::CompoundConfig> config;
  problem::Workload workload;
  model::Engine::Specs arch_specs;
  std::unique_ptr<mapspace::MapSpace> mapspace;
  std::vector<mapspace::MapSpace*> split_mapspaces;
  sparse::SparseOptimizationInfo sparse_opts;
  std::vector<std::unique_ptr<search::SearchAlgorithm>> search_algs;

  uint64_t num_threads;
  std::vector<std::string> opt_metrics;
  unsigned int search_size;
  unsigned int timeout;
  unsigned int victory_cond;

  void SetUp(const ::benchmark::State& state) {
    config = std::make_unique<config::CompoundConfig>(config_yaml_, "yaml");
    auto root_node = config->getRoot();

    auto problem_node = root_node.lookup("problem");
    problem::ParseWorkload(problem_node, workload);

    config::CompoundConfigNode arch;
    if (root_node.exists("arch")) {
      arch = root_node.lookup("arch");
    } else if (root_node.exists("architecture")) {
      arch = root_node.lookup("architecture");
    }
    arch_specs = model::Engine::ParseSpecs(arch);

    num_threads = state.range(0);
    opt_metrics = {"edp"};
    search_size = 1000;
    timeout = 500;
    victory_cond = 500;

    config::CompoundConfigNode arch_constraints, mapspace_cfg;
    if (arch.exists("constraints")) {
      arch_constraints = arch.lookup("constraints");
    } else if (root_node.exists("arch_constraints")) {
      arch_constraints = root_node.lookup("arch_constraints");
    } else if (root_node.exists("architecture_constraints")) {
      arch_constraints = root_node.lookup("architecture_constraints");
    }

    if (root_node.exists("mapspace")) {
      mapspace_cfg = root_node.lookup("mapspace");
    } else if (root_node.exists("mapspace_constraints")) {
      mapspace_cfg = root_node.lookup("mapspace_constraints");
    }

    // Ignore cout
    std::cout.setstate(std::ios_base::failbit);

    mapspace = std::unique_ptr<mapspace::MapSpace>(mapspace::ParseAndConstruct(
        mapspace_cfg, arch_constraints, arch_specs, workload));
    split_mapspaces = mapspace->Split(num_threads);

    auto search_cfg = root_node.lookup("mapper");
    for (unsigned t = 0; t < num_threads; t++) {
      search_algs.emplace_back(
          search::ParseAndConstruct(search_cfg, split_mapspaces[t], t));
    }

    config::CompoundConfigNode sparse_opts_cfg;
    if (root_node.exists("sparse_optimizations")) {
      sparse_opts_cfg = root_node.lookup("sparse_optimizations");
    }
    sparse_opts = sparse::ParseAndConstruct(sparse_opts_cfg, arch_specs);

    // Stop ignoring cout
    std::cout.clear();
  }

 protected:
  std::string config_yaml_ =
      "\
architecture:\n\
  version: 0.2\n\
\n\
  subtree:\n\
  - name: System\n\
\n\
    local:\n\
    - name: MainMemory\n\
      class: DRAM\n\
      attributes:\n\
        width: 256\n\
        block-size: 32\n\
        word-bits: 8\n\
\n\
    subtree:\n\
    - name: Chip\n\
      attributes:\n\
        technology: 40nm\n\
\n\
      local:\n\
      - name: GlobalBuffer # 256KB buffer\n\
        class: SRAM\n\
        attributes:\n\
          depth: 8192\n\
          width: 256\n\
          block-size: 32\n\
          word-bits: 8\n\
\n\
      subtree:\n\
      - name: PE\n\
\n\
        local:\n\
        - name: RegisterFile\n\
          class: regfile\n\
          attributes:\n\
            depth: 64\n\
            width: 8\n\
            block-size: 1\n\
            word-bits: 8\n\
        - name: MACC\n\
          class: intmac\n\
          attributes:\n\
            datawidth: 8\n\
\n\
mapspace_constraints:\n\
  - target: MainMemory\n\
    type: temporal\n\
    factors: R=1 P=1 K=1\n\
    permutation: PRK\n\
\n\
  - target: GlobalBuffer\n\
    type: temporal\n\
    factors: R=1 P=16 K=16\n\
    permutation: PRK\n\
\n\
  - target: RegisterFile\n\
    type: temporal\n\
    factors: R=3 P=1 K=2\n\
    permutation: RPK\n\
\n\
  - target: GlobalBuffer\n\
    type: bypass\n\
    keep:\n\
    - Weights\n\
    - Inputs\n\
    - Outputs\n\
\n\
  - target: RegisterFile\n\
    type: bypass\n\
    keep:\n\
    - Weights\n\
    - Inputs\n\
    - Outputs\n\
\n\
mapper:\n\
  optimization-metric: [ delay, energy ]\n\
  num-threads: 1\n\
  algorithm: linear-pruned\n\
  victory-condition: 0\n\
  timeout: 0\n\
\n\
problem:\n\
  shape:\n\
    name: Conv1D_OC\n\
    dimensions: [ K, R, P ]\n\
    data-spaces:\n\
    - name: Weights\n\
      projection:\n\
      - [ [K] ]\n\
      - [ [R] ]\n\
    - name: Inputs\n\
      projection:\n\
      - [ [R], [P] ]\n\
    - name: Outputs\n\
      projection:\n\
      - [ [K] ]\n\
      - [ [P] ]\n\
      read-write: True\n\
\n\
  instance:\n\
    K: 32\n\
    R: 3\n\
    P: 16\n\
\n";
};

BENCHMARK_DEFINE_F(ThreeLevelArchMapping, DecoupledMapperTest)
(benchmark::State& state) {
  std::vector<search::SearchAlgorithm*> search_alg_ptrs;
  for (auto& s : search_algs) {
    search_alg_ptrs.push_back(s.get());
  }

  DecoupledMapper mapper(arch_specs, workload, split_mapspaces, search_alg_ptrs,
                         sparse_opts, opt_metrics, search_algs.size());

  for (auto _ : state) {
    mapper.Run();
  }
}

BENCHMARK_REGISTER_F(ThreeLevelArchMapping, DecoupledMapperTest)
    ->RangeMultiplier(2)
    ->Range(1, 8);
