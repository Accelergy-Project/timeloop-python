#pragma once

#include <compound-config/compound-config.hpp>
#include <mapping/arch-properties.hpp>
#include <mapping/constraints.hpp>
#include <mapspaces/mapspace-base.hpp>
#include <model/sparse-optimization-info.hpp>
#include <optional>
#include <string>

namespace pytimeloop::configurator {
class Configurator {
 public:
  static Configurator FromYamlStr(const std::string& config_str);

  std::optional<ArchProperties> GetArchProperties();
  std::optional<model::Engine::Specs> GetArchSpecs();
  std::optional<Mapping> GetMapping();
  std::optional<mapping::Constraints> GetMappingConstraints();
  std::optional<sparse::SparseOptimizationInfo> GetSparseOptimizations();
  problem::Workload& GetWorkload();

 private:
  std::unique_ptr<config::CompoundConfig> config_;

  std::unique_ptr<ArchProperties> arch_props_;
  std::unique_ptr<model::Engine::Specs> arch_specs_;
  std::unique_ptr<Mapping> mapping_;
  std::unique_ptr<mapping::Constraints> mapping_constraints_;
  std::unique_ptr<mapspace::MapSpace> mapspace_;
  std::unique_ptr<sparse::SparseOptimizationInfo> sparse_opts_;
  std::unique_ptr<problem::Workload> workload_;

  Configurator();

  void ParseConfig();
};
};  // namespace pytimeloop::configurator