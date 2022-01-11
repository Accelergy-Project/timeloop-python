cmake_minimum_required(VERSION 3.4)
project(timeloop_python)

find_package(benchmark REQUIRED)

if(DEFINED ENV{LIBTIMELOOP_PATH})
    set(LIBTIMELOOP_PATH $ENV{LIBTIMELOOP_PATH})
else()
    set(LIBTIMELOOP_PATH ${PROJECT_SOURCE_DIR}/lib/timeloop)
endif()
message(STATUS "Using Timeloop library from: ${LIBTIMELOOP_PATH}")

find_library(LIBTIMELOOP timeloop-model HINTS ${LIBTIMELOOP_PATH}/lib REQUIRED)

add_library(define_gterminate SHARED bindings/gterminate.cpp)

set(PYTIMELOOP_SRC
    src/mapper/decoupled-mapper.cpp
    src/model/accelerator.cpp
    src/model/accelerator-pool.cpp)
set(PYTIMELOOP_HDR
    include/pytimeloop/mapper/decoupled-mapper.h
    include/pytimeloop/mapper/mapper-base.h
    include/pytimeloop/model/accelerator-pool.h
    include/pytimeloop/model/accelerator.h
    include/pytimeloop/model/eval-result.h
    include/pytimeloop/search/mapspace-search.h)
add_library(pytimeloop SHARED ${PYTIMELOOP_SRC} ${PYTIMELOOP_HDR})
set_property(TARGET pytimeloop PROPERTY CXX_STANDARD 17)
target_include_directories(pytimeloop PRIVATE
    include
    ${LIBTIMELOOP_PATH}/include
    lib/pybind11/include)
target_link_libraries(pytimeloop PRIVATE ${LIBTIMELOOP})

set(BINDINGS_SRC 
    bindings/accelergy_interface.cpp
    bindings/bindings.cpp
    bindings/config.cpp
    bindings/mapper.cpp
    bindings/mapping.cpp
    bindings/mapspace.cpp
    bindings/model.cpp
    bindings/problem.cpp
    bindings/search.cpp)
add_subdirectory(lib/pybind11)
pybind11_add_module(bindings ${BINDINGS_SRC})
set_property(TARGET bindings PROPERTY CXX_STANDARD 17)
target_include_directories(bindings PRIVATE
    include
    ${LIBTIMELOOP_PATH}/include
    lib/pybind11/include)
target_link_libraries(bindings PRIVATE config++ yaml-cpp ${LIBTIMELOOP} define_gterminate pytimeloop)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(bindings PRIVATE VERSION_INFO=${VERSION_INFO})

set(BENCHMARK_SRC
    benchmark/bench-main.cpp)
add_executable(pytimeloop_bench ${BENCHMARK_SRC})
set_property(TARGET pytimeloop_bench PROPERTY CXX_STANDARD 17)
target_include_directories(pytimeloop_bench PRIVATE
    include
    ${LIBTIMELOOP_PATH}/include
    lib/pybind11/include)
target_link_libraries(pytimeloop_bench PRIVATE benchmark::benchmark pytimeloop define_gterminate ${LIBTIMELOOP} config++ yaml-cpp)
