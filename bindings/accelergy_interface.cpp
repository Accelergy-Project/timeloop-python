#include "pytimeloop/bindings/accelergy-interface.h"

// Timeloop headers
#include "util/accelergy_interface.hpp"

namespace pytimeloop::accelergy_bindings {

void BindAccelergyInterface(py::module &m) {
  m.def("native_invoke_accelergy", &accelergy::invokeAccelergy,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>(),
        "Invokes Accelergy");
}

}  // namespace pytimeloop::accelergy_bindings
