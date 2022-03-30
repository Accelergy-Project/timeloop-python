#pragma once

#include <memory>
#include <optional>

// Timeloop library
#include <mapping/mapping.hpp>
#include <mapspaces/mapspace-base.hpp>
#include <search/search.hpp>

namespace pytimeloop::pysearch {

class MapSpaceSearchAlgorithm {
 public:
  typedef std::reference_wrapper<MapSpaceSearchAlgorithm> Ref;
  typedef std::shared_ptr<MapSpaceSearchAlgorithm> ShPtr;

  struct NextMapping {
    Mapping mapping;
    bool only_bypass;
  };

  virtual std::optional<NextMapping> Next() = 0;
  virtual void Report(search::Status status, double cost = 0) = 0;
};

class TimeloopSearchAlgorithm : public MapSpaceSearchAlgorithm {
 public:
  using MapSpaceSearchAlgorithm::NextMapping;

  TimeloopSearchAlgorithm(search::SearchAlgorithm& alg,
                          mapspace::MapSpace& mapspace)
      : alg_(alg), mapspace_(mapspace) {}

  std::optional<NextMapping> Next() override {
    mapspace::ID mapping_id;
    if (alg_.Next(mapping_id)) {
      Mapping mapping;
      auto status = mapspace_.ConstructMapping(mapping_id, &mapping);
      bool success =
          std::accumulate(status.begin(), status.end(), true,
                          [](bool cur, const mapspace::Status& status) {
                            return cur && status.success;
                          });
      if (success) {
        bool only_bypass = true;
        for (unsigned idim = 0; idim < unsigned(mapspace::Dimension::Num);
             idim++) {
          if (mapspace::Dimension(idim) != mapspace::Dimension::DatatypeBypass)
            only_bypass &= (mapping_id[idim] == prev_mapping_id_[idim]);
        }
        prev_mapping_id_ = std::move(mapping_id);

        return NextMapping{std::move(mapping), only_bypass};
      } else {
        alg_.Report(search::Status::MappingConstructionFailure);
      }
    }
    return std::nullopt;
  }

  void Report(search::Status status, double cost = 0) override {
    alg_.Report(status, cost);
  }

 private:
  search::SearchAlgorithm& alg_;
  mapspace::MapSpace& mapspace_;
  mapspace::ID prev_mapping_id_;
};

}  // namespace pytimeloop::pysearch
