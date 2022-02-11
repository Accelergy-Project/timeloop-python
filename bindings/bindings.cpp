#include <variant>
#include <vector>

#include "pytimeloop/bindings/accelergy-interface.h"
#include "pytimeloop/bindings/config.h"
#include "pytimeloop/bindings/mapper.h"
#include "pytimeloop/bindings/mapping.h"
#include "pytimeloop/bindings/mapspace.h"
#include "pytimeloop/bindings/model.h"
#include "pytimeloop/bindings/problem.h"
#include "pytimeloop/bindings/search.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(bindings, m) {
  m.doc() = "PyTimeloop bindings to C++ timeloop code ";

  BindAccelergyInterface(m);
  BindConfigClasses(m);

  mapper_bindings::BindDecoupledMapper(m);

  BindMappingClasses(m);
  BindMapspaceClasses(m);

  model_bindings::BindAccelerator(m);
  model_bindings::BindAcceleratorPool(m);
  model_bindings::BindEngine(m);
  model_bindings::BindEvaluationResult(m);
  model_bindings::BindLevel(m);
  model_bindings::BindSparseOptimizationInfo(m);
  model_bindings::BindTopology(m);

  BindProblemClasses(m);
  BindSearchClasses(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
