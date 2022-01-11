#pragma once

#include <benchmark/benchmark.h>

#include "pytimeloop/model/accelerator.h"

// Timeloop
#include <compound-config/compound-config.hpp>
#include <mapping/parser.hpp>
#include <model/sparse-optimization-parser.hpp>
#include <workload/workload.hpp>

class OneLevelArch : public benchmark::Fixture {
 public:
  problem::Workload workload_;
  model::Engine::Specs arch_specs_;
  Mapping mapping_;
  sparse::SparseOptimizationInfo sparse_opts_;

  void SetUp(const ::benchmark::State& state) {
    config::CompoundConfig config(config_yaml_, "yaml");
    auto root_node = config.getRoot();

    auto problem_node = root_node.lookup("problem");
    problem::ParseWorkload(problem_node, workload_);

    config::CompoundConfigNode arch;
    if (root_node.exists("arch")) {
      arch = root_node.lookup("arch");
    } else if (root_node.exists("architecture")) {
      arch = root_node.lookup("architecture");
    }
    arch_specs_ = model::Engine::ParseSpecs(arch);

    auto mapping_node = root_node.lookup("mapping");
    mapping_ = mapping::ParseAndConstruct(mapping_node, arch_specs_, workload_);

    config::CompoundConfigNode sparse_node;
    if (root_node.exists("sparse_optimizations")) {
      sparse_node = root_node.lookup("sparse_optimizations");
    }
    // Ignore cout from parsing
    std::cout.setstate(std::ios_base::failbit);
    sparse_opts_ = sparse::ParseAndConstruct(sparse_node, arch_specs_);
    std::cout.clear();
  }

 private:
  const std::string arch_yaml_ =
      "\
architecture:\n\
  version: 0.2\n\
\n\
  subtree:\n\
  - name: PE\n\
    attributes:\n\
      technology: 40nm\n\
\n\
    local:\n\
    - name: Buffer\n\
      class: regfile\n\
      attributes:\n\
        depth: 64\n\
        width: 8\n\
        block-size: 1\n\
        word-bits: 8\n\
\n\
    - name: MACC\n\
      class: intmac\n\
      attributes:\n\
        datawidth: 8\n";

  const std::string map_yaml_ =
      "\
mapping:\n\
  - target: Buffer\n\
    type: temporal\n\
    factors: R=3 P=16\n\
    permutation: RP\n";

  const std::string workload_yaml_ =
      "\
problem:\n\
  shape:\n\
    name: Conv1D\n\
    dimensions: [ R, P ]\n\
    data-spaces:\n\
    - name: Weights\n\
      projection:\n\
      - [ [R] ]\n\
    - name: Inputs\n\
      projection:\n\
      - [ [R], [P] ]\n\
    - name: Outputs\n\
      projection:\n\
      - [ [P] ]\n\
      read-write: True\n\
\n\
  instance:\n\
    R: 3\n\
    P: 16\n";

  const std::string config_yaml_ = arch_yaml_ + map_yaml_ + workload_yaml_;
};

BENCHMARK_F(OneLevelArch, AcceleratorTest)(benchmark::State& state) {
  Accelerator acc(arch_specs_);

  for (auto _ : state) {
    auto res = acc.Evaluate(mapping_, workload_, sparse_opts_);
  }
}
