cmake_minimum_required(VERSION 3.4)
project(timeloop_python)

add_subdirectory(pybind11)
pybind11_add_module(bindings bindings/bindings.cpp)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(bindings PRIVATE VERSION_INFO=${VERSION_INFO})
