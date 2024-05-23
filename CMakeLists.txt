cmake_minimum_required(VERSION 3.4)
project(timeloop_python)

if(DEFINED ENV{TIMELOOP_INCLUDE_PATH})
    message(STATUS "Using timeloop includes from: $ENV{TIMELOOP_INCLUDE_PATH}")
    set(TIMELOOP_INCLUDE_PATH $ENV{TIMELOOP_INCLUDE_PATH})
else()
    message(FATAL_ERROR "Please point TIMELOOP_INCLUDE_PATH to your Timeloop headers.")
endif()

if(DEFINED ENV{TIMELOOP_LIB_PATH})
    set(TIMELOOP_LIB_PATH $ENV{TIMELOOP_LIB_PATH})
    message(STATUS "Using timeloop libraries from: $ENV{TIMELOOP_LIB_PATH}")
    find_library(LIBTIMELOOP timeloop-mapper HINTS ${TIMELOOP_LIB_PATH} REQUIRED)
else()
    find_library(LIBTIMELOOP timeloop-mapper REQUIRED)
endif()

find_package(Threads REQUIRED)

add_library(define_gterminate STATIC bindings/gterminate.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(PYTIMELOOP_SRC
    src/mapper/coupled-mapper.cpp
    src/mapspace/status.cpp
    src/model/accelerator.cpp
    src/model/bounded-queue-accelerator-pool.cpp
    src/model/unbounded-queue-accelerator-pool.cpp
    src/model/util.cpp)
set(PYTIMELOOP_HDR
    include/pytimeloop/mapper/coupled-mapper.h
    include/pytimeloop/mapper/mapper-base.h
    include/pytimeloop/model/accelerator-pool.h
    include/pytimeloop/model/accelerator.h
    include/pytimeloop/model/eval-result.h
    include/pytimeloop/model/util.h
    include/pytimeloop/search/mapspace-search.h)
add_library(pytimeloop SHARED ${PYTIMELOOP_SRC} ${PYTIMELOOP_HDR})
set_property(TARGET pytimeloop PROPERTY CXX_STANDARD 17)
target_include_directories(pytimeloop PRIVATE
    include
    ${TIMELOOP_INCLUDE_PATH}
    lib/pybind11/include)
target_link_libraries(pytimeloop PRIVATE ${LIBTIMELOOP} define_gterminate)

set(BINDINGS_SRC 
    src/configurator/configurator.cpp
    src/mapper/coupled-mapper.cpp
    src/mapspace/status.cpp
    src/model/accelerator.cpp
    src/model/bounded-queue-accelerator-pool.cpp
    src/model/unbounded-queue-accelerator-pool.cpp
    src/model/util.cpp
    bindings/accelergy_interface.cpp
    bindings/bindings.cpp
    bindings/buffer.cpp
    bindings/config.cpp
    bindings/mapper.cpp
    bindings/mapping.cpp
    bindings/mapspace.cpp
    bindings/model.cpp
    bindings/problem.cpp
    bindings/search.cpp
    bindings/topology.cpp)
add_subdirectory(lib/pybind11)
pybind11_add_module(bindings ${BINDINGS_SRC})
set_property(TARGET bindings PROPERTY CXX_STANDARD 17)
target_include_directories(bindings PUBLIC
    include
    ${TIMELOOP_INCLUDE_PATH}
    lib/pybind11/include)
target_link_libraries(bindings PUBLIC config++ yaml-cpp rt ${LIBTIMELOOP} define_gterminate)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(bindings PRIVATE VERSION_INFO=${VERSION_INFO})
