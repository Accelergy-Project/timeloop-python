#include "pytimeloop/mapper/decoupled-mapper.h"

#include "util.h"

namespace pytimeloop::pymapper {

DecoupledMapper::DecoupledMapper(
    const ArchSpecs& arch_spec, const Workload& workload,
    std::vector<MapSpace*>& mapspaces,
    std::vector<SearchAlgorithm*>& search_algs,
    const SparseOptInfo& sparse_opts, const std::vector<std::string>& metrics,
    unsigned acc_pool_nthreads, uint64_t search_size, unsigned timeout,
    unsigned victory_condition, bool penalize_consecutive_bypass_fails)
    : arch_spec_(arch_spec),
      workload_(workload),
      sparse_opts_(sparse_opts),
      metrics_(metrics),
      pool_nthreads_(acc_pool_nthreads),
      search_size_(search_size),
      timeout_(timeout),
      victory_cond_(victory_condition),
      penalize_cons_bypass_fails_(penalize_consecutive_bypass_fails) {
  if (search_size_ > 0) {
    search_size_ = 1 + (search_size_ - 1) / search_algs_.size();
  }

  assert(mapspaces.size() == search_algs.size());
  search_algs_.reserve(mapspaces.size());
  for (unsigned i = 0; i < mapspaces.size(); i++) {
    search_algs_.emplace_back(std::make_shared<TimeloopSearchAlgorithm>(
        *search_algs[i], *mapspaces[i]));
  }
}

DecoupledMapper::DecoupledMapper(
    const ArchSpecs& arch_spec, const Workload& workload,
    std::vector<MapSpaceSearchAlgorithm::ShPtr>& search_algs,
    const SparseOptInfo& sparse_opts, const std::vector<std::string>& metrics,
    unsigned acc_pool_nthreads, uint128_t search_size, unsigned timeout,
    unsigned victory_condition, bool penalize_consecutive_bypass_fails)
    : arch_spec_(arch_spec),
      workload_(workload),
      sparse_opts_(sparse_opts),
      metrics_(metrics),
      pool_nthreads_(acc_pool_nthreads),
      search_size_(search_size),
      timeout_(timeout),
      victory_cond_(victory_condition),
      penalize_cons_bypass_fails_(penalize_consecutive_bypass_fails) {
  if (search_size_ > 0) {
    search_size_ = 1 + (search_size_ - 1) / search_algs_.size();
  }

  search_algs_.reserve(search_algs.size());
  for (auto alg : search_algs) {
    search_algs_.emplace_back(alg);
  }
}

Mapping DecoupledMapper::Run() {
  BoundedQueueAcceleratorPool pool(arch_spec_, pool_nthreads_,
                                   2 * pool_nthreads_);

  std::vector<std::optional<SearchTask>> outstanding_tasks;
  for (auto& alg : search_algs_) {
    outstanding_tasks.push_back(SearchSendNext(alg, pool));
  }

  while (true) {
    bool all_terminated = true;
    for (auto& alg : search_algs_) {
      all_terminated = all_terminated && alg.terminated;
    }
    if (all_terminated) {
      break;
    }

    auto result = pool.GetResult();
    bool found = false;
    for (unsigned i = 0; i < outstanding_tasks.size(); i++) {
      auto& task_opt = outstanding_tasks[i];
      if (!task_opt) continue;

      auto& task = *task_opt;
      auto& alg = search_algs_.at(i);
      if (!alg.terminated && task.task_id == result.id) {
        SearchReport(alg, result, task);
        outstanding_tasks[i] = SearchSendNext(alg, pool);
        found = true;
        break;
      }
    }
    assert(found);
  }

  return best_mapping_;
}

std::optional<DecoupledMapper::SearchTask> DecoupledMapper::SearchSendNext(
    MapperSearchAlgorithm& alg, AcceleratorPool& pool) {
  if (search_size_ > 0 && alg.valid_maps == search_size_) alg.terminated = true;

  if (victory_cond_ > 0 && alg.maps_since_last_best == victory_cond_)
    alg.terminated = true;

  if (alg.invld_maps_mapcnstr + alg.invld_maps_eval > 0 &&
      alg.invld_maps_mapcnstr + alg.invld_maps_eval == timeout_)
    alg.terminated = true;

  // Get next mapping from search algorithm
  auto next_mapping_opt = alg.alg->Next();
  alg.total_maps++;
  if (!next_mapping_opt) {
    alg.terminated = true;
    alg.invld_maps_mapcnstr++;
  }

  if (alg.terminated) {
    return std::nullopt;
  }

  // Pre-evaluation and evaluation
  auto& mapping = (*next_mapping_opt).mapping;
  bool only_bypass = (*next_mapping_opt).only_bypass;
  auto task_id = pool.Evaluate(mapping, workload_, sparse_opts_, false);

  return DecoupledMapper::SearchTask{task_id, std::move(mapping), only_bypass};
}

void DecoupledMapper::SearchReport(MapperSearchAlgorithm& alg,
                                   EvaluationResult& result,
                                   DecoupledMapper::SearchTask& task) {
  if (!result.eval_status) {
    if (penalize_cons_bypass_fails_ || !task.only_bypass) {
      alg.invld_maps_eval++;
    }
    alg.alg->Report(search::Status::EvalFailure);
    return;
  }

  // Evaluated
  for (model::EvalStatus& res : *(result.eval_status)) {
    if (!res.success) {
      if (penalize_cons_bypass_fails_ || !task.only_bypass) {
        alg.invld_maps_eval++;
      }
      alg.alg->Report(search::Status::EvalFailure);
      return;
    }
  }

  // Success!
  alg.valid_maps++;
  alg.invld_maps_mapcnstr = 0;
  alg.invld_maps_eval = 0;
  alg.alg->Report(search::Status::Success, Cost(result, metrics_[0]));
  if (IsBetter(result, best_result_, metrics_)) {
    best_result_ = result;
    best_mapping_ = task.mapping;
    alg.maps_since_last_best = 0;
  } else {
    if (penalize_cons_bypass_fails_ || !task.only_bypass) {
      alg.maps_since_last_best++;
    }
  }
}

}  // namespace pytimeloop::pymapper
