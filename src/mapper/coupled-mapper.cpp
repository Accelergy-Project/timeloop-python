#include "pytimeloop/mapper/coupled-mapper.h"

#include "pytimeloop/model/util.h"

namespace pytimeloop::pymapper {

CoupledMapper::CoupledMapper(
    const ArchSpecs& arch_spec, Workload& workload,
    std::vector<std::pair<MapSpace*, SearchAlgorithm*>>&
        mapspace_search_alg_pairs,
    SparseOptInfo& sparse_opts, const std::vector<std::string>& metrics,
    uint64_t search_size, unsigned timeout, unsigned victory_condition,
    bool penalize_consecutive_bypass_fails)
    : arch_spec_(arch_spec),
      workload_(workload),
      mapspace_search_alg_pairs_(mapspace_search_alg_pairs),
      sparse_opts_(sparse_opts),
      metrics_(metrics),
      search_size_(search_size),
      timeout_(timeout),
      victory_cond_(victory_condition),
      penalize_cons_bypass_fails_(penalize_consecutive_bypass_fails),
      submapper_pool_() {
  submapper_pool_ = std::make_unique<WorkerPool<SubMapSpaceMapper>>(
      mapspace_search_alg_pairs_.size(), [&]() {
        return SubMapSpaceMapper(arch_spec_, workload_, sparse_opts_, metrics_,
                                 search_size, timeout, victory_condition,
                                 penalize_consecutive_bypass_fails);
      });
}

std::pair<Mapping, EvaluationResult> CoupledMapper::Run() {
  for (auto& [mapspace, search_alg] : mapspace_search_alg_pairs_) {
    submapper_pool_->PushTask({mapspace, search_alg});
  }

  auto result = submapper_pool_->PopResult();
  auto best_mapping = result.val.mapping;
  auto best_result = result.val.eval_result;

  for (unsigned i = 1; i < mapspace_search_alg_pairs_.size(); ++i) {
    auto result = submapper_pool_->PopResult();
    if (pytimeloop::pymodel::IsBetter(result.val.eval_result, best_result,
                                      metrics_)) {
      best_mapping = result.val.mapping;
    }
  }

  submapper_pool_->Terminate();

  return {best_mapping, best_result};
}

CoupledMapper::SubMapSpaceMapper::SubMapSpaceMapper(
    const ArchSpecs& arch_spec, Workload& workload, SparseOptInfo& sparse_opts,
    const std::vector<std::string>& metrics, uint128_t search_size,
    unsigned timeout, unsigned victory_condition,
    bool penalize_consecutive_bypass_fails)
    : arch_spec(arch_spec),
      workload(workload),
      sparse_opts(sparse_opts),
      metrics(metrics),
      search_size(search_size),
      timeout(timeout),
      victory_condition(victory_condition),
      penalize_consecutive_bypass_fails(penalize_consecutive_bypass_fails) {}

CoupledMapper::SubMapSpaceResult CoupledMapper::SubMapSpaceMapper::operator()(
    Task& task) {
  auto mapspace = task.first;
  auto search_alg = task.second;

  Accelerator acc(arch_spec);

  mapspace::ID prev_mapping_id;
  bool terminate = false;

  while (!terminate) {
    if (search_size > 0 && valid_mappings == search_size) {
      terminate = true;
    }

    if (victory_condition > 0 &&
        mappings_since_last_best == victory_condition) {
      terminate = true;
    }

    if ((invld_mappings_mapcnstr + invld_mappings_eval) > 0 &&
        (invld_mappings_mapcnstr + invld_mappings_eval) == timeout) {
      terminate = true;
    }

    mapspace::ID mapping_id;
    if (!search_alg->Next(mapping_id)) {
      terminate = true;
    }

    if (terminate) {
      break;
    }

    // TODO sync thread best to global best

    // Check if the only change is in the Bypass dimension.
    bool only_bypass_changed = false;
    if (total_mappings > 1) {
      bool match = true;
      for (unsigned idim = 0; idim < unsigned(mapspace::Dimension::Num);
           ++idim) {
        if (mapspace::Dimension(idim) != mapspace::Dimension::DatatypeBypass) {
          match &= (mapping_id[idim] == prev_mapping_id[idim]);
        }
      }
      only_bypass_changed = match;
    }
    prev_mapping_id = mapping_id;

    bool success = true;

    // Stage 1: Construct mapping from mapping ID.
    Mapping mapping;

    auto construction_status = mapspace->ConstructMapping(mapping_id, &mapping);
    success &=
        std::accumulate(construction_status.begin(), construction_status.end(),
                        true, [](bool cur, const mapspace::Status& status) {
                          return cur && status.success;
                        });

    total_mappings++;

    if (!success) {
      invld_mappings_mapcnstr++;
      search_alg->Report(search::Status::MappingConstructionFailure);
      continue;
    }

    // Stage 2 & 3
    auto result = acc.Evaluate(mapping, workload, sparse_opts);

    success &= result.eval_status.has_value();
    if (!success) {
      if (penalize_consecutive_bypass_fails || !only_bypass_changed) {
        invld_mappings_eval++;
      }
      search_alg->Report(search::Status::EvalFailure);
      continue;
    }

    auto& status_per_level = result.eval_status.value();
    success &=
        std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                        [](bool cur, const model::EvalStatus& status) {
                          return cur && status.success;
                        });
    if (!success) {
      if (penalize_consecutive_bypass_fails || !only_bypass_changed) {
        invld_mappings_eval++;
      }
      search_alg->Report(search::Status::EvalFailure);
      continue;
    }

    // Success!!
    valid_mappings++;
    invld_mappings_mapcnstr = 0;
    invld_mappings_eval = 0;

    search_alg->Report(search::Status::Success,
                       pytimeloop::pymodel::Cost(result, metrics.at(0)));

    if (pytimeloop::pymodel::IsBetter(result, best_result, metrics)) {
      best_result = result;
      best_mapping = mapping;
      mappings_since_last_best = 0;
    } else if (penalize_consecutive_bypass_fails || !only_bypass_changed) {
      mappings_since_last_best++;
    }
  }

  return SubMapSpaceResult{.mapping = best_mapping, .eval_result = best_result};
}

}  // namespace pytimeloop::pymapper
