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
        /// @brief Creates an equivalent BufferLevel class in Python.
        using BufferLevel = model::BufferLevel;
        py::class_<BufferLevel> bufferLevel(m, "BufferLevel");

        bufferLevel
            .def(py::init<>());

        /// @brief Binds BufferLevel::Stats to BufferLevel.Stats in Python.
        using Stats = BufferLevel::Stats;
        py::class_<Stats>(bufferLevel, "Stats");
    }
}