#pragma once

// PyBind11 headers
#include "pybind11/pybind11.h"

namespace pytimeloop::looptree_bindings
{

void BindLooptree(pybind11::module& m);

}
