#include "bindings.h"

#include <variant>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(bindings, m) {
  m.doc() = R"pbdoc(
        PyTimeloop bindings to C++ timeloop code
        -----------------------
        .. currentmodule:: pytimeloop
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

  BindAccelergyInterface(m);
  BindConfigClasses(m);
  BindMappingClasses(m);
  BindModelClasses(m);
  BindProblemClasses(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
