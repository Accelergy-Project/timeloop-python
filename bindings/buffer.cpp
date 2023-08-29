// PyTimeloop header files.
#include "pytimeloop/bindings/buffer.h"
#include "pytimeloop/bindings/type_casters.h"

// Timeloop headers
#include "model/buffer.hpp"

namespace pytimeloop::model_bindings
{
void BindBufferClasses(py::module& m)
{
/**
 * @brief           Binds the BufferLevel class to Python under Topology
 * @warning         May break once the global assumptions of Workload no
 *                  longer are true.
 */
py::class_<model::BufferLevel> buffer_level(m, "BufferLevel");

buffer_level.doc() = R"DOCSTRING(
    @brief  BufferLevel is a class that represents a single level of a memory 
            hierarchy. It is used by Topology to represent a BufferLevel.
)DOCSTRING";

buffer_level
    .def(py::init<>())
    /// @brief Uses BufferLevel's stream export for the string represntation.
    .def("__str__", [](const model::BufferLevel& self) 
    {
        std::stringstream stream;
        stream << self;
        return stream.str();
    },
    R"DOCSTRING(
        @brief  Returns a string representation of the BufferLevel in Timeloop's
                pretty printed format.
    )DOCSTRING")
    /// @brief Read-only methods to access the fields of BufferLevel.
    .def_property_readonly("name", &model::BufferLevel::Name)
    /// @todo .def_property_readonly("specs", &model::BufferLevel::GetSpecs)
    .def_property_readonly("stats", &model::BufferLevel::GetStats);


/**
 * @brief               Binds the Specs of BufferLevel to Python under BufferLevel.
 * @param buffer_level  Making Specs under the scope of BufferLevel.
 * @warning             May break once the global assumptions of Workload no
 *                     longer are true.
 */
/**
 * @todo: Uncomment this and .def_property_readonly("specs", &model::BufferLevel::GetSpecs)
 * once Attribute<type> is bound to python.
py::class_<model::BufferLevel::Specs> specs(buffer_level, "Specs");
specs.doc() = R"DOCSTRING(
    @brief  The specifications of a BufferLevel.
)DOCSTRING";

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
    .def_readonly("reduction_supported", &model::BufferLevel::Specs::reduction_supported)
    .def_readonly("network_fill_latency", &model::BufferLevel::Specs::network_fill_latency)
    .def_readonly("network_drain_latency", &model::BufferLevel::Specs::network_drain_latency)
    .def_readonly("concordant_compressed_tile_traversal", &model::BufferLevel::Specs::concordant_compressed_tile_traversal)
    .def_readonly("tile_partition_supported", &model::BufferLevel::Specs::tile_partition_supported)
    .def_readonly("decompression_supported", &model::BufferLevel::Specs::decompression_supported)
    .def_readonly("compression_supported", &model::BufferLevel::Specs::compression_supported)
    .def_readonly("metadata_storage_width", &model::BufferLevel::Specs::metadata_storage_width)
    .def_readonly("metadata_storage_depth", &model::BufferLevel::Specs::metadata_storage_depth)
    .def_readonly("unified_data_md_storage", &model::BufferLevel::Specs::unified_data_md_storage)
    .def_readonly("default_md_block_size", &model::BufferLevel::Specs::default_md_block_size)
    .def_readonly("default_md_word_bits", &model::BufferLevel::Specs::default_md_word_bits)
    .def_readonly("read_network_name", &model::BufferLevel::Specs::read_network_name)
    .def_readonly("fill_network_name", &model::BufferLevel::Specs::fill_network_name)
    .def_readonly("drain_network_name", &model::BufferLevel::Specs::drain_network_name)
    .def_readonly("update_network_name", &model::BufferLevel::Specs::update_network_name)
    .def_readonly("ERT_entries", &model::BufferLevel::Specs::ERT_entries)
    .def_readonly("op_energy_map", &model::BufferLevel::Specs::op_energy_map)
    .def_readonly("allow_overbooking", &model::BufferLevel::Specs::allow_overbooking)
    .def_readonly("vector_access_energy", &model::BufferLevel::Specs::vector_access_energy)
    .def_readonly("storage_area", &model::BufferLevel::Specs::storage_area)
    .def_readonly("addr_gen_energy", &model::BufferLevel::Specs::addr_gen_energy)
    .def_readonly("access_energy_source", &model::BufferLevel::Specs::access_energy_source)
    .def_readonly("addr_gen_energy_source", &model::BufferLevel::Specs::addr_gen_energy_source)
    .def_readonly("storage_area_source", &model::BufferLevel::Specs::storage_area_source)
    .def_readonly("is_sparse_module", &model::BufferLevel::Specs::is_sparse_module);
*/


/**
 * @brief                   Binds BufferLevel::Stats to BufferLevel.Stats in Python.
 * @param   buffer_level    Making BufferLevel.Stats under the scope of BufferLevel.
 * @warning                 May break once the global assumptions of Workload no
 *                         `longer are true.
 */
py::class_<model::BufferLevel::Stats> stats(buffer_level, "Stats");

stats.doc() = R"DOCSTRING(
    @brief  The statistics of a BufferLevel.
)DOCSTRING";

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
    .def_readonly("energy_due_to_overflow", &model::BufferLevel::Stats::energy_due_to_overflow)
    .def_readonly("tile_shape", &model::BufferLevel::Stats::tile_shape)
    .def_readonly("data_tile_size", &model::BufferLevel::Stats::data_tile_size)
    .def_readonly("compressed", &model::BufferLevel::Stats::compressed)
    .def_readonly("metadata_tile_size", &model::BufferLevel::Stats::metadata_tile_size)
    .def_readonly("metadata_tile_size_bits", &model::BufferLevel::Stats::metadata_tile_size_bits)
    .def_readonly("metadata_format", &model::BufferLevel::Stats::metadata_format)
    .def_readonly("tile_confidence", &model::BufferLevel::Stats::tile_confidence)
    .def_readonly("parent_level_name", &model::BufferLevel::Stats::parent_level_name)
    .def_readonly("parent_level_id", &model::BufferLevel::Stats::parent_level_id)
    .def_readonly("tile_density_distribution", &model::BufferLevel::Stats::tile_density_distribution)
    .def_readonly("format_shared_bandwidth_ratio", &model::BufferLevel::Stats::format_shared_bandwidth_ratio)
    .def_readonly("format_read_bandwidth_ratio", &model::BufferLevel::Stats::format_read_bandwidth_ratio)
    .def_readonly("format_write_bandwidth_ratio", &model::BufferLevel::Stats::format_write_bandwidth_ratio)
    .def_readonly("fine_grained_scalar_accesses", &model::BufferLevel::Stats::fine_grained_scalar_accesses)
    .def_readonly("fine_grained_format_scalar_accesses", &model::BufferLevel::Stats::fine_grained_format_scalar_accesses)
    .def_readonly("fine_grained_vector_accesses", &model::BufferLevel::Stats::fine_grained_vector_accesses)
    .def_readonly("fine_grained_fromat_accesses_bits", &model::BufferLevel::Stats::fine_grained_fromat_accesses_bits)
    .def_readonly("gated_reads", &model::BufferLevel::Stats::gated_reads)
    .def_readonly("skipped_reads", &model::BufferLevel::Stats::skipped_reads)
    .def_readonly("random_reads", &model::BufferLevel::Stats::random_reads)
    .def_readonly("gated_fills", &model::BufferLevel::Stats::gated_fills)
    .def_readonly("skipped_fills", &model::BufferLevel::Stats::skipped_fills)
    .def_readonly("random_fills", &model::BufferLevel::Stats::random_fills)
    .def_readonly("gated_updates", &model::BufferLevel::Stats::gated_updates)
    .def_readonly("skipped_updates", &model::BufferLevel::Stats::skipped_updates)
    .def_readonly("random_updates", &model::BufferLevel::Stats::random_updates)
    .def_readonly("random_format_reads", &model::BufferLevel::Stats::random_format_reads)
    .def_readonly("skipped_format_reads", &model::BufferLevel::Stats::skipped_format_reads)
    .def_readonly("gated_format_reads", &model::BufferLevel::Stats::gated_format_reads)
    .def_readonly("random_format_fills", &model::BufferLevel::Stats::random_format_fills)
    .def_readonly("skipped_format_fills", &model::BufferLevel::Stats::skipped_format_fills)
    .def_readonly("gated_format_fills", &model::BufferLevel::Stats::gated_format_fills)
    .def_readonly("random_format_updates", &model::BufferLevel::Stats::random_format_updates)
    .def_readonly("skipped_format_updates", &model::BufferLevel::Stats::skipped_format_updates)
    .def_readonly("gated_format_updates", &model::BufferLevel::Stats::gated_format_updates)
    .def_readonly("decompression_counts", &model::BufferLevel::Stats::decompression_counts)
    .def_readonly("compression_counts", &model::BufferLevel::Stats::compression_counts)
    .def_readonly("cycles", &model::BufferLevel::Stats::cycles)
    .def_readonly("slowdown", &model::BufferLevel::Stats::slowdown);
}
} // namespace pytimeloop::model_bindings