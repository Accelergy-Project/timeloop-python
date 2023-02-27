#include "pytimeloop/configurator/configurator.h"

#include "mapping/parser.hpp"
#include "model/sparse-optimization-parser.hpp"
#include "mapspaces/mapspace-factory.hpp"

namespace pytimeloop::configurator {

Configurator Configurator::FromYamlStr(const std::string& config_str) {
  auto configurator = Configurator();
  configurator.config_ =
      std::make_unique<config::CompoundConfig>(config_str, "yaml");

  configurator.ParseConfig();

  return configurator;
}

template <typename T>
inline std::optional<T> CopyOrNull(std::unique_ptr<T>& obj) {
  if (obj) {
    return *obj;
  }
  return std::nullopt;
}

std::optional<ArchProperties> Configurator::GetArchProperties() {
  return CopyOrNull(arch_props_);
}

std::optional<model::Engine::Specs> Configurator::GetArchSpecs() {
  return CopyOrNull(arch_specs_);
}

std::optional<Mapping> Configurator::GetMapping() {
  return CopyOrNull(mapping_);
}

std::optional<mapping::Constraints> Configurator::GetMappingConstraints() {
  return CopyOrNull(mapping_constraints_);
}

std::optional<sparse::SparseOptimizationInfo>
Configurator::GetSparseOptimizations() {
  return CopyOrNull(sparse_opts_);
}

problem::Workload& Configurator::GetWorkload() {
  return *workload_;
}

Configurator::Configurator()
    : config_(nullptr),
      workload_(std::make_unique<problem::Workload>()),
      arch_specs_(std::make_unique<model::Engine::Specs>()) {}

void Configurator::ParseConfig() {
  auto root_node = config_->getRoot();

  workload_ = std::make_unique<problem::Workload>();
  problem::ParseWorkload(root_node.lookup("problem"), *workload_);

  config::CompoundConfigNode arch;
  if (root_node.exists("arch")) {
    arch = root_node.lookup("arch");
  } else if (root_node.exists("architecture")) {
    arch = root_node.lookup("architecture");
  }

  bool is_sparse_topology = root_node.exists("sparse_optimizations");
  *arch_specs_ = model::Engine::ParseSpecs(arch, is_sparse_topology);

  if (root_node.exists("ERT")) {
    arch_specs_->topology.ParseAccelergyERT(root_node.lookup("ERT"));
    arch_specs_->topology.ParseAccelergyART(root_node.lookup("ART"));
  }

  config::CompoundConfigNode sparse_opts_config;
  if (is_sparse_topology) {
    sparse_opts_config = root_node.lookup("sparse_optimizations");
  }
  sparse_opts_ = std::make_unique<sparse::SparseOptimizationInfo>(
      sparse::ParseAndConstruct(sparse_opts_config, *arch_specs_));

  workload_->SetDefaultDenseTensorFlag(
      sparse_opts_->compression_info.all_ranks_default_dense);

  arch_props_ = std::make_unique<ArchProperties>(*arch_specs_);

  config::CompoundConfigNode arch_constraints;
  if (arch.exists("constraints")) {
    arch_constraints = arch.lookup("constraints");
  } else if (root_node.exists("arch_constraints")) {
    arch_constraints = root_node.lookup("arch_constraints");
  } else if (root_node.exists("architecture_constraints")) {
    arch_constraints = root_node.lookup("architecture_constraints");
  }

  mapping_constraints_ =
      std::make_unique<mapping::Constraints>(*arch_props_, *workload_);
  mapping_constraints_->Parse(arch_constraints);

  mapping_ = std::make_unique<Mapping>(mapping::ParseAndConstruct(
      root_node.lookup("mapping"), *arch_specs_, *workload_));

  config::CompoundConfigNode mapspace;
  if (root_node.exists("mapspace")) {
    mapspace = root_node.lookup("mapspace");
  } else if (root_node.exists("mapspace_constraints")) {
    mapspace = root_node.lookup("mapspace_constraints");
  }

  auto filter_spatial_fanout =
    sparse_opts_->action_spatial_skipping_info.size() == 0;
  mapspace_ =
    std::unique_ptr<mapspace::MapSpace>(mapspace::ParseAndConstruct(
      mapspace,
      arch_constraints,
      *arch_specs_,
      *workload_,
      filter_spatial_fanout
    ));
}

};  // namespace pytimeloop::configurator