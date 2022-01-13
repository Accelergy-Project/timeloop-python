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
  std::unique_ptr<config::CompoundConfig> config;
  problem::Workload workload;
  model::Engine::Specs arch_specs;
  Mapping mapping;
  sparse::SparseOptimizationInfo sparse_opts;

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

    auto mapping_node = root_node.lookup("mapping");
    mapping = mapping::ParseAndConstruct(mapping_node, arch_specs, workload);

    config::CompoundConfigNode sparse_node;
    if (root_node.exists("sparse_optimizations")) {
      sparse_node = root_node.lookup("sparse_optimizations");
    }
    // Ignore cout from parsing
    std::cout.setstate(std::ios_base::failbit);
    sparse_opts = sparse::ParseAndConstruct(sparse_node, arch_specs);
    std::cout.clear();
  }

 protected:
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

  const std::string workloadyaml_ =
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

  const std::string config_yaml_ = arch_yaml_ + map_yaml_ + workloadyaml_;
};

BENCHMARK_F(OneLevelArch, AcceleratorTest)(benchmark::State& state) {
  Accelerator acc(arch_specs);

  for (auto _ : state) {
    auto res = acc.Evaluate(mapping, workload, sparse_opts);
  }
}

BENCHMARK_F(OneLevelArch, PreEvaluation)(benchmark::State& state) {
  model::Engine engine;
  engine.Spec(arch_specs);

  for (auto _ : state) {
    auto pre_eval_status =
        engine.PreEvaluationCheck(mapping, workload, &sparse_opts, false);
    bool success =
        std::accumulate(pre_eval_status.begin(), pre_eval_status.end(), true,
                        [](bool cur, const model::EvalStatus& status) {
                          return cur && status.success;
                        });
    if (!success) {
      auto res = EvaluationResult{.pre_eval_status = pre_eval_status,
                                  .eval_status = std::nullopt};
    }
  }
}

BENCHMARK_F(OneLevelArch, Evaluation)(benchmark::State& state) {
  model::Engine engine;
  engine.Spec(arch_specs);
  auto pre_eval_status =
      engine.PreEvaluationCheck(mapping, workload, &sparse_opts, false);
  bool success =
      std::accumulate(pre_eval_status.begin(), pre_eval_status.end(), true,
                      [](bool cur, const model::EvalStatus& status) {
                        return cur && status.success;
                      });

  for (auto _ : state) {
    auto eval_status = engine.Evaluate(mapping, workload, &sparse_opts);
    for (unsigned level = 0; level < eval_status.size(); level++) {
      if (!eval_status[level].success) {
        std::cerr << "ERROR: couldn't map level\n";
      }
    }
  }
}

BENCHMARK_F(OneLevelArch, ReturnResult)(benchmark::State& state) {
  model::Engine engine;
  engine.Spec(arch_specs);
  auto pre_eval_status =
      engine.PreEvaluationCheck(mapping, workload, &sparse_opts, false);
  bool success =
      std::accumulate(pre_eval_status.begin(), pre_eval_status.end(), true,
                      [](bool cur, const model::EvalStatus& status) {
                        return cur && status.success;
                      });

  auto eval_status = engine.Evaluate(mapping, workload, &sparse_opts);

  for (auto _ : state) {
    if (engine.IsEvaluated()) {
      auto topology = engine.GetTopology();
      auto res = EvaluationResult{0,
                                  pre_eval_status,
                                  eval_status,
                                  engine.Utilization(),
                                  engine.Energy(),
                                  engine.Area(),
                                  engine.Cycles(),
                                  topology.AlgorithmicComputes(),
                                  topology.ActualComputes(),
                                  topology.LastLevelAccesses()};
    } else {
      auto res = EvaluationResult{0, pre_eval_status, std::nullopt};
    }
  }
}
