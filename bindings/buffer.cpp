#include "pytimeloop/bindings/buffer.h"

#include <optional>
#include <variant>

// PyBind11 headers
#include "pybind11/stl.h"

// Timeloop headers
#include "model/buffer.hpp"

namespace pytimeloop::buffer_bindings
{
void BindBufferClasses(py::module& m)
{
/**
 * @brief           Binds the BufferLevel class to Python under Topology
 * @param topology  Making BufferLevel under the scope of Topology.
 * @warning         May break once the global assumptions of Workload no
 *                  longer are true.
 */
py::class_<model::BufferLevel> buffer_level(m, "BufferLevel");

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
py::class_<model::BufferLevel::Specs> specs(buffer_level, "Specs");

specs
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



/**
 * @brief                   Binds BufferLevel::Stats to BufferLevel.Stats in Python.
 * @param   buffer_level    Making BufferLevel.Stats under the scope of BufferLevel.
 * @warning                 May break once the global assumptions of Workload no
 *                         longer are true.
 */
py::class_<model::BufferLevel::Stats> stats(buffer_level, "Stats");

stats
    .def(py::init<>())
    /// @brief Exposes all stats attributes as read-only to Python.
    .def_readonly("keep", &model::BufferLevel::Stats::keep)
    .def_readonly("partition_size", &model::BufferLevel::Stats::partition_size)
    .def_readonly("utilized_capacity", &model::BufferLevel::Stats::utilized_capacity)
    .def_readonly("utilized_md_capacity_bits", &model::BufferLevel::Stats::utilized_md_capacity_bits)
    .def_readonly("tile_size", &model::BufferLevel::Stats::tile_size)
    .def_readonly("utilized_instances", &model::BufferLevel::Stats::utilized_instances)
    .def_readonly("utilized_x_expansion", &model::BufferLevel::Stats::utilized_x_expansion)
    .def_readonly("utilized_y_expansion", &model::BufferLevel::Stats::utilized_y_expansion)
    .def_readonly("utilized_clusters", &model::BufferLevel::Stats::utilized_clusters)
    .def_readonly("reads", &model::BufferLevel::Stats::reads)
    .def_readonly("updates", &model::BufferLevel::Stats::updates)
    .def_readonly("fills", &model::BufferLevel::Stats::fills)
    .def_readonly("address_generations", &model::BufferLevel::Stats::address_generations)
    .def_readonly("temporal_reductions", &model::BufferLevel::Stats::temporal_reductions)
    .def_readonly("shared_bandwidth", &model::BufferLevel::Stats::shared_bandwidth)
    .def_readonly("read_bandwidth", &model::BufferLevel::Stats::read_bandwidth)
    .def_readonly("write_bandwidth", &model::BufferLevel::Stats::write_bandwidth)
    .def_readonly("energy_per_algorithmic_access", &model::BufferLevel::Stats::energy_per_algorithmic_access)
    .def_readonly("energy_per_access", &model::BufferLevel::Stats::energy_per_access)
    .def_readonly("energy", &model::BufferLevel::Stats::energy)
    .def_readonly("temporal_reduction_energy", &model::BufferLevel::Stats::temporal_reduction_energy)
    .def_readonly("addr_gen_energy", &model::BufferLevel::Stats::addr_gen_energy)
    .def_readonly("cluster_access_energy", &model::BufferLevel::Stats::cluster_access_energy)
    .def_readonly("cluster_access_energy_due_to_overflow", &model::BufferLevel::Stats::cluster_access_energy_due_to_overflow)
    .def_readonly("energy_due_to_overflow", &model::BufferLevel::Stats::energy_due_to_overflow);
}
}