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
        using Stats = model::BufferLevel::Stats;
        py::class_<Stats>(m, "Stats");
    }
}