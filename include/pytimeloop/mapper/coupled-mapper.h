#pragma once

#include <utility>
#include <vector>

// Timeloop library
#include <mapping/mapping.hpp>
#include <model/engine.hpp>
#include <search/search.hpp>

#include "pytimeloop/mapper/mapper-base.h"
#include "pytimeloop/model/accelerator.h"
#include "pytimeloop/utils/worker-pool.h"

namespace pytimeloop::pymapper {

using namespace pytimeloop::pymodel;
using namespace pytimeloop::pysearch;

class CoupledMapper : public Mapper {
 public:
  CoupledMapper(const ArchSpecs& arch_spec, Workload& workload,
                std::vector<std::pair<MapSpace*, SearchAlgorithm*>>&
                    mapspace_search_alg_pairs,
                SparseOptInfo& sparse_opts,
                const std::vector<std::string>& metrics,
                uint64_t search_size = 0, unsigned timeout = 500,
                unsigned victory_condition = 500,
                bool penalize_consecutive_bypass_fails = false);

  std::pair<Mapping, EvaluationResult> Run();

 private:
  struct SubMapSpaceResult {
    Mapping mapping;
    EvaluationResult eval_result;
  };

  struct SubMapSpaceMapper {
    typedef std::pair<MapSpace*, SearchAlgorithm*> Task;
    typedef SubMapSpaceResult Result;

    const ArchSpecs& arch_spec;
    Workload& workload;
    SparseOptInfo& sparse_opts;
    const std::vector<std::string>& metrics;

    uint128_t search_size;
    unsigned timeout;
    unsigned victory_condition;
    bool penalize_consecutive_bypass_fails;

    uint128_t total_mappings = 0;
    uint128_t valid_mappings = 0;
    uint128_t invld_mappings_mapcnstr = 0;
    uint128_t invld_mappings_eval = 0;
    unsigned mappings_since_last_best = 0;

    EvaluationResult best_result;
    Mapping best_mapping;

    SubMapSpaceMapper(const ArchSpecs& arch_spec, Workload& workload,
                      SparseOptInfo& sparse_opts,
                      const std::vector<std::string>& metrics,
                      uint128_t search_size, unsigned timeout,
                      unsigned victory_condition,
                      bool penalize_consecutive_bypass_fails);

    Result operator()(Task& task);
  };

 private:
  const ArchSpecs& arch_spec_;
  Workload& workload_;
  const std::vector<std::pair<MapSpace*, SearchAlgorithm*>>
      mapspace_search_alg_pairs_;
  SparseOptInfo& sparse_opts_;
  const std::vector<std::string> metrics_;
  uint128_t search_size_;
  unsigned timeout_;
  unsigned victory_cond_;
  unsigned penalize_cons_bypass_fails_;

  std::unique_ptr<WorkerPool<SubMapSpaceMapper>> submapper_pool_;
};

}  // namespace pytimeloop::pymapper
