#include <variant>
#include <vector>

#include "pytimeloop/bindings/accelergy-interface.h"
#include "pytimeloop/bindings/buffer.h"
#include "pytimeloop/bindings/config.h"
#include "pytimeloop/bindings/mapper.h"
#include "pytimeloop/bindings/mapping.h"
#include "pytimeloop/bindings/mapspace.h"
#include "pytimeloop/bindings/model.h"
#include "pytimeloop/bindings/problem.h"
#include "pytimeloop/bindings/search.h"
#include "pytimeloop/bindings/topology.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(bindings, m) {
  using namespace pytimeloop;

  m.doc() = R"DOCSTRING(
    @brief PyTimeloop bindings to C++ timeloop code.
  )DOCSTRING";

  auto accelergy_submodule = m.def_submodule("accelergy");
  accelergy_bindings::BindAccelergyInterface(accelergy_submodule);

  auto config_submodule = m.def_submodule("config");
  config_submodule.doc() = R"DOCSTRING(
    @brief The configuration classes needed to run Timeloop in Python
  )DOCSTRING";
  config_bindings::BindConfigClasses(config_submodule);

  auto mapper_submodule = m.def_submodule("mapper");
  mapper_bindings::BindDecoupledMapper(mapper_submodule);

  auto mapping_submodule = m.def_submodule("mapping");
  mapping_bindings::BindMappingClasses(mapping_submodule);

  auto mapspace_submodule = m.def_submodule("mapspace");
  mapspace_bindings::BindMapspaceClasses(mapspace_submodule);

  auto model_submodule = m.def_submodule("model");
  model_submodule.doc() = R"DOCSTRING(
    @brief  The classes neeeded to build a model in Timeloop.
  )DOCSTRING";
  model_bindings::BindAccelerator(model_submodule);
  model_bindings::BindBufferClasses(model_submodule);
  model_bindings::BindAcceleratorPool(model_submodule);
  model_bindings::BindEngine(model_submodule);
  model_bindings::BindEvaluationResult(model_submodule);
  model_bindings::BindLevel(model_submodule);
  model_bindings::BindSparseOptimizationInfo(model_submodule);
  model_bindings::BindTopology(model_submodule);

  auto problem_submodule = m.def_submodule("problem");
  problem_bindings::BindProblemClasses(problem_submodule);

  auto search_submodule = m.def_submodule("search");
  search_bindings::BindSearchClasses(search_submodule);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
