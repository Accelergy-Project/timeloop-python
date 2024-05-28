#pragma once

// PyBind11 headers
#include "pybind11/pybind11.h"

namespace pytimeloop::application_bindings
{

void BindApplications(pybind11::module& m);

}
