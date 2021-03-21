cmake_minimum_required(VERSION 3.4)
project(timeloop_python)

find_library(LIBTIMELOOP timeloop-model HINTS ${PROJECT_SOURCE_DIR}/timeloop/lib)

add_library(define_gterminate SHARED bindings/gterminate.cpp)

set(BINDINGS_SRC 
    bindings/bindings.cpp bindings/accelergy_interface.cpp bindings/config.cpp
    bindings/mapping.cpp bindings/model.cpp bindings/problem.cpp)
add_subdirectory(pybind11)
pybind11_add_module(bindings ${BINDINGS_SRC})
set_property(TARGET bindings PROPERTY CXX_STANDARD 17)
target_include_directories(bindings PRIVATE timeloop/src pybind11/include)
target_link_libraries(bindings PRIVATE config++ yaml-cpp ${LIBTIMELOOP} define_gterminate)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(bindings PRIVATE VERSION_INFO=${VERSION_INFO})
