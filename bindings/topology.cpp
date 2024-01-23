// Necessary for PerDataSpace to be exported.
#include "pytimeloop/bindings/topology.h"
#include "pytimeloop/bindings/type_casters.h"

// Timeloop headers
#include "model/topology.hpp"

namespace pytimeloop::model_bindings
{
void BindTopology(py::module& m) {
  /**
   * @brief   Binds the model::Topology class to Python.
   * @param m The module we're binding Topology to.
   */
  py::class_<model::Topology> topology(m, "Topology");

  topology.doc() = R"DOCSTRING(
      @brief Binds the Timeloop Topology class to Python.
      @note  This class is a wrapper around Timeloop's native Topology class.
  )DOCSTRING";

  topology
      /// @brief Uses the stream export of topology for the string representation.
      .def("__str__", [](const model::Topology& self) 
      {
        std::stringstream stream;
        stream << self;
        return stream.str();
      },
      R"DOCSTRING(
        @brief  Returns the Timeloop pretty printed version of Topology.
      )DOCSTRING")
      /// @brief General attributes of Topology.
      .def_property_readonly("algorithmic_computes", &model::Topology::AlgorithmicComputes)
      .def_property_readonly("actual_computes", &model::Topology::ActualComputes)
      .def_property_readonly("last_level_accesses", &model::Topology::LastLevelAccesses)
      .def_property_readonly("tile_sizes", &model::Topology::TileSizes)
      .def_property_readonly("utilized_capacities", &model::Topology::UtilizedCapacities)
      /// @brief Retrieves the stats from self.
      .def_property_readonly("stats", &model::Topology::GetStats)
      /// @brief Retrieves the buffer levels from self.
      .def_property_readonly("buffer_levels", [](const model::Topology& self) {
        /* Creates a vector of levels, as self.NumStorageLevels() cannot be used
         * in constexpr, meaning we can't use it to initialize array sizes. */
        std::vector<std::shared_ptr<const model::BufferLevel>> levels(self.NumStorageLevels());

        // Goes through all the levels and creates the pairs.
        for (std::uint32_t index = 0; index < self.NumStorageLevels(); index++)
        {
          // Creates the pair of the level's name and the level itself.
          levels[index] = self.ViewStorageLevel(index);
        }

        // Returns the array of levels.
        return levels;
      });

  // Shorthand of Stats for brevity.
  using Stats = model::Topology::Stats;
  /** 
   * @brief Binds the Stats of Topology to Python
   * 
   * @todo  Figure out if C++ has compile time Reflection (specifically 
   *        Introspection) in order to make this less ugly and spaghetti code.
   * 
   * @param topology  Making Stats under the scope of Topology.
   */
  py::class_<Stats> stats(topology, "Stats");

  stats.doc() = R"DOCSTRING(
    @brief  Binds the Topology Stats struct to Python.
    @note   This class is a wrapper around Timeloop's native Stats struct.
  )DOCSTRING";
      
  stats
      /// @brief The reset function built into the Stats struct.
      .def("reset", &Stats::Reset)
      /// @brief Read-only methods to access the fields of stats.
      .def_readonly("energy", &Stats::energy)
      .def_readonly("area", &Stats::area)
      .def_readonly("cycles", &Stats::cycles)
      .def_readonly("utilization", &Stats::utilization)
      /// @note BEGIN PerDataSpace INTERNALS
      .def_readonly("tile_sizes", &Stats::tile_sizes)
      .def_readonly("utilized_capacities", &Stats::utilized_capacities)
      .def_readonly("utilized_instances", &Stats::utilized_instances)
      /// @note END PerDataSpace INTERNALS
      .def_readonly("algorithmic_computes", &Stats::algorithmic_computes)
      .def_readonly("actual_computes", &Stats::actual_computes)
      .def_readonly("last_level_accesses", &Stats::last_level_accesses)
      .def_readonly("accesses", &Stats::accesses)
      .def_readonly("per_tensor_accesses", &Stats::per_tensor_accesses);
}
}   // namespace pytimeloop::model_bindings