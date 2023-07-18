#include "pytimeloop/bindings/model.h"

#include "pytimeloop/model/accelerator-pool.h"
#include "pytimeloop/model/accelerator.h"

#include <array>
#include <utility>
#include <string>

// PyBind11 headers
#include <pybind11/stl.h>

// Timeloop headers
#include "model/engine.hpp"
#include "model/level.hpp"
#include "model/sparse-optimization-info.hpp"
#include "model/sparse-optimization-parser.hpp"
#include "workload/util/per-data-space.hpp"

namespace pytimeloop::model_bindings {
using namespace pytimeloop::pymodel;

void BindAccelerator(py::module& m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init<const model::Engine::Specs&>())
      .def("evaluate", &Accelerator::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>());
}

void BindAcceleratorPool(py::module& m) {
  py::class_<UnboundedQueueAcceleratorPool>(m, "UnboundedAcceleratorPool")
      .def(py::init<const model::Engine::Specs&, unsigned>())
      .def("evaluate", &UnboundedQueueAcceleratorPool::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("get_result", &UnboundedQueueAcceleratorPool::GetResult);

  py::class_<BoundedQueueAcceleratorPool>(m, "BoundedAcceleratorPool")
      .def(py::init<const model::Engine::Specs&, size_t, size_t>())
      .def("evaluate", &BoundedQueueAcceleratorPool::Evaluate,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("get_result", &BoundedQueueAcceleratorPool::GetResult);
}

void BindEngine(py::module& m) {
  py::class_<model::Engine::Specs>(m, "ArchSpecs")
      .def(py::init(&model::Engine::ParseSpecs))
      .def_static("parse_specs", &model::Engine::ParseSpecs,
                  "Parse architecture specifications.")
      .def("parse_accelergy_art",
           [](model::Engine::Specs& s, config::CompoundConfigNode& art) {
             s.topology.ParseAccelergyART(art);
           })
      .def(
          "parse_accelergy_ert",
          [](model::Engine::Specs& s, config::CompoundConfigNode& ert) {
            s.topology.ParseAccelergyERT(ert);
          },
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>())
      .def("level_names",
           [](model::Engine::Specs& s) { return s.topology.LevelNames(); })
      .def("storage_level_names", [](model::Engine::Specs& s) {
        return s.topology.StorageLevelNames();
      });

  py::class_<model::Engine>(m, "Engine")
      .def(py::init<>(),
           "Construct wrapper over Timeloop's native Engine class. Consider "
           "using `pytimeloop.Accelerator` instead. \n"
           "Engine.spec has to be called later with ArchSpecs.")
      .def(py::init([](model::Engine::Specs specs) {
             auto e = std::make_unique<model::Engine>();
             e->Spec(specs);
             return e;
           }),
           "Construct and spec Engine.")
      .def("spec", &model::Engine::Spec)
      .def("pre_evaluation_check", &model::Engine::PreEvaluationCheck)
      .def("evaluate", &model::Engine::Evaluate, py::arg("mapping"),
           py::arg("workload"), py::arg("sparse_optimizations"),
           py::arg("break_on_failure") = true,
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
      .def("is_evaluated", &model::Engine::IsEvaluated)
      .def("utilization", &model::Engine::Utilization)
      .def("energy", &model::Engine::Energy)
      .def("area", &model::Engine::Area)
      .def("cycles", &model::Engine::Cycles)
      .def("get_topology", &model::Engine::GetTopology)
      .def("pretty_print_stats", [](model::Engine& e) -> std::string {
        std::stringstream ss;
        ss << e << std::endl;
        return ss.str();
      });
}

void BindEvaluationResult(py::module& m) {
  py::class_<EvaluationResult>(m, "EvaluationResult")
      .def_readonly("id", &EvaluationResult::id)
      .def_readonly("pre_eval_status", &EvaluationResult::pre_eval_status)
      .def_readonly("eval_status", &EvaluationResult::eval_status)
      .def_readonly("utilization", &EvaluationResult::utilization)
      .def_readonly("energy", &EvaluationResult::energy)
      .def_readonly("area", &EvaluationResult::area)
      .def_readonly("cycles", &EvaluationResult::cycles)
      .def_readonly("algorithmic_computes",
                    &EvaluationResult::algorithmic_computes)
      .def_readonly("actual_computes", &EvaluationResult::actual_computes)
      .def_readonly("last_level_accesses",
                    &EvaluationResult::last_level_accesses)
      .def(py::pickle(
        [](const EvaluationResult& e) {
          return py::make_tuple(e.id,
                                e.pre_eval_status,
                                e.eval_status,
                                e.utilization,
                                e.energy,
                                e.area,
                                e.cycles,
                                e.algorithmic_computes,
                                e.actual_computes,
                                e.last_level_accesses);
        },
        [](py::tuple t) {
          return EvaluationResult{
            .id = t[0].cast<uint64_t>(),
            .pre_eval_status = t[1].cast<std::vector<model::EvalStatus>>(),
            .eval_status =
              t[2].cast<std::optional<std::vector<model::EvalStatus>>>(),
            .utilization = t[3].cast<double>(),
            .energy = t[4].cast<double>(),
            .area = t[5].cast<double>(),
            .cycles = t[6].cast<uint64_t>(),
            .algorithmic_computes = t[7].cast<uint64_t>(),
            .actual_computes = t[8].cast<uint64_t>(),
            .last_level_accesses = t[9].cast<uint64_t>()};
        }
      ));
}

void BindLevel(py::module& m) {
  py::class_<model::EvalStatus>(m, "EvalStatus")
      .def_readonly("success", &model::EvalStatus::success)
      .def_readonly("fail_reason", &model::EvalStatus::fail_reason)
      .def("__repr__", [](model::EvalStatus& e) -> std::string {
        if (e.success) {
          return "EvalStatus(success=1)";
        } else {
          return "EvalStatus(success=0, fail_reason=" + e.fail_reason + ")";
        }
      });
}

void BindSparseOptimizationInfo(py::module& m) {
  py::class_<sparse::SparseOptimizationInfo>(m, "SparseOptimizationInfo")
      .def(py::init(&sparse::ParseAndConstruct))
      .def_static("parse_and_construct", &sparse::ParseAndConstruct,
                  py::call_guard<py::scoped_ostream_redirect,
                                 py::scoped_estream_redirect>());
}

void BindTopology(py::module& m) {

  /**
   * @brief   Binds the model::Topology class to Python.
   * @param m The module we're binding Topology to.
   */
  py::class_<model::Topology> topology(m, "Topology");

  topology
      /// @brief Uses the stream export of topology for the string representation.
      .def("__str__", [](const model::Topology& self) 
      {
        std::stringstream stream;
        stream << self;
        return stream.str();
      })
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
        std::vector<model::BufferLevel> levels(self.NumStorageLevels());

        // Goes through all the levels and creates the pairs.
        for (std::uint32_t index = 0; index < self.NumStorageLevels(); index++)
        {
          // Creates the pair of the level's name and the level itself.
          levels[index] = self.ExposeStorageLevel(index);
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

  /**
   * @brief   Binds PerDataSpace as used in Topology.Stats to Python under
   *          Topology.Stats.
   * @warning May break once the global assumptions of PerDataSpace no longer
   *          are true.
   */
  using PerDataSpace = problem::PerDataSpace<std::uint64_t>;
  py::class_<PerDataSpace>(stats, "PerDataSpace")
      .def(py::init<>())
      /// @brief Index accession of the Array.
      .def("__getitem__", [](const PerDataSpace& self, const long long& index)
      {
        return self[index];
      })
      /** @brief Takes advantage of the built-in PerDataSpace streamer to output
       * a string. */
      .def("__repr__", [](const PerDataSpace& self) 
      {
        std::stringstream stream;
        stream << self;
        return stream.str();
      });
  /**
   * @brief           Binds the BufferLevel class to Python under Topology
   * @param topology  Making BufferLevel under the scope of Topology.
   * @warning         May break once the global assumptions of Workload no
   *                 longer are true.
   */
  py::class_<model::BufferLevel> buffer_level(topology, "BufferLevel");

  buffer_level
      .def(py::init<>())
      /// @brief Uses BufferLevel's stream export for the string represntation.
      .def("__str__", [](const model::BufferLevel& self) 
      {
        std::stringstream stream;
        stream << self;
        return stream.str();
      })
      /// @brief Read-only methods to access the fields of BufferLevel.
      .def_property_readonly("name", &model::BufferLevel::Name)
      .def_property_readonly("specs", &model::BufferLevel::GetSpecs)
      .def_property_readonly("stats", &model::BufferLevel::GetStats);
  
  /**
   * @brief               Binds the Specs of BufferLevel to Python under BufferLevel.
   * @param buffer_level  Making Specs under the scope of BufferLevel.
   * @warning             May break once the global assumptions of Workload no
   *                     longer are true.
   */
  py::class_<model::BufferLevel::Specs> buffer_level_specs(buffer_level, "Specs");

  buffer_level_specs
      .def(py::init<>())
      /// @brief Exposes all specs attributes as read-only to Python.
      .def_readonly("name", &model::BufferLevel::Specs::name)
      .def_readonly("technology", &model::BufferLevel::Specs::technology)
      .def_readonly("size", &model::BufferLevel::Specs::size)
      .def_readonly("md_size", &model::BufferLevel::Specs::md_size)
      .def_readonly("md_size_bits", &model::BufferLevel::Specs::md_size_bits)
      .def_readonly("word_bits", &model::BufferLevel::Specs::word_bits)
      .def_readonly("addr_gen_bits", &model::BufferLevel::Specs::addr_gen_bits)
      .def_readonly("block_size", &model::BufferLevel::Specs::block_size)
      .def_readonly("cluster_size", &model::BufferLevel::Specs::cluster_size)
      .def_readonly("instances", &model::BufferLevel::Specs::instances)
      .def_readonly("meshX", &model::BufferLevel::Specs::meshX)
      .def_readonly("meshY", &model::BufferLevel::Specs::meshY)
      .def_readonly("shared_bandwidth", &model::BufferLevel::Specs::shared_bandwidth)
      .def_readonly("read_bandwidth", &model::BufferLevel::Specs::read_bandwidth)
      .def_readonly("write_bandwidth", &model::BufferLevel::Specs::write_bandwidth)
      .def_readonly("multiple_buffering", &model::BufferLevel::Specs::multiple_buffering)
      .def_readonly("effective_size", &model::BufferLevel::Specs::effective_size)
      .def_readonly("effective_md_size", &model::BufferLevel::Specs::effective_md_size)
      .def_readonly("effective_md_size_bits", &model::BufferLevel::Specs::effective_md_size_bits)
      .def_readonly("min_utilization", &model::BufferLevel::Specs::min_utilization)
      .def_readonly("num_ports", &model::BufferLevel::Specs::num_ports)
      .def_readonly("num_banks", &model::BufferLevel::Specs::num_banks)
      .def_readonly("reduction_supported", &model::BufferLevel::Specs::reduction_supported);



}
}  // namespace pytimeloop::model_bindings

