set(TEST_SRC
    test-main.cpp)

set(TEST_HDR
    test-accelerator.h
    test-concurrent-queue.h
    test-mapper.h
    test-worker-pool.h)

add_executable(pytimeloop_cpptest ${TEST_SRC} ${TEST_HDR})

set_property(TARGET pytimeloop_cpptest PROPERTY CXX_STANDARD 17)

target_include_directories(pytimeloop_cpptest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TIMELOOP_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/lib/pybind11/include)

target_link_libraries(pytimeloop_cpptest PRIVATE
    ${LIBTIMELOOP}
    config++
    yaml-cpp
    rt
    define_gterminate
    pytimeloop
    Threads::Threads)

add_test(pytimeloop_cpptest pytimeloop_cpptest)
