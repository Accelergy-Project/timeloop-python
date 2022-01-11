#pragma once

#include <memory>
#include <optional>
#include <vector>

// Timeloop library
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <search/search.hpp>

#include "pytimeloop/mapper/mapper-base.h"
#include "pytimeloop/model/accelerator-pool.h"
#include "pytimeloop/search/mapspace-search.h"

class DecoupledMapper : public Mapper {
 public:
  DecoupledMapper(const ArchSpecs& arch_spec, const Workload& workload,
                  std::vector<MapSpace*>& mapspaces,
                  std::vector<SearchAlgorithm*>& search_algs,
                  const SparseOptInfo& sparse_opts,
                  const std::vector<std::string>& metrics,
                  unsigned acc_pool_nthreads, uint64_t search_size = 0,
                  unsigned timeout = 500, unsigned victory_condition = 500,
                  bool penalize_consecutive_bypass_fails = false);

  DecoupledMapper(const ArchSpecs& arch_spec, const Workload& workload,
                  std::vector<MapSpaceSearchAlgorithm::ShPtr>& search_algs,
                  const SparseOptInfo& sparse_opts,
                  const std::vector<std::string>& metrics,
                  unsigned acc_pool_nthreads, uint128_t search_size = 0,
                  unsigned timeout = 500, unsigned victory_condition = 500,
                  bool penalize_consecutive_bypass_fails = false);

  Mapping Run();

 private:
  const ArchSpecs& arch_spec_;
  const Workload& workload_;
  const SparseOptInfo& sparse_opts_;
  const std::vector<std::string> metrics_;
  unsigned pool_nthreads_;
  uint128_t search_size_;
  unsigned timeout_;
  unsigned victory_cond_;
  unsigned penalize_cons_bypass_fails_;

  struct MapperSearchAlgorithm {
    MapSpaceSearchAlgorithm::ShPtr alg;
    uint64_t total_maps;
    uint64_t valid_maps;
    uint64_t invld_maps_mapcnstr;
    uint64_t invld_maps_eval;
    uint64_t maps_since_last_best;
    bool terminated;

    MapperSearchAlgorithm(MapSpaceSearchAlgorithm::ShPtr alg)
        : alg(std::move(alg)),
          total_maps(0),
          valid_maps(0),
          invld_maps_mapcnstr(0),
          invld_maps_eval(0),
          maps_since_last_best(0),
          terminated(false) {}
  };

  struct SearchTask {
    uint64_t task_id;
    Mapping mapping;
    bool only_bypass;
  };

  EvaluationResult best_result_;
  Mapping best_mapping_;
  std::vector<MapperSearchAlgorithm> search_algs_;

  std::optional<SearchTask> SearchSendNext(MapperSearchAlgorithm& alg,
                                           AcceleratorPool& pool);
  void SearchReport(MapperSearchAlgorithm& alg, EvaluationResult& result,
                    SearchTask& task);
};
