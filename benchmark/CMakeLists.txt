find_package(benchmark REQUIRED)

set(BENCHMARK_SRC                                                           
	bench-main.cpp)                                               
add_executable(pytimeloop_bench ${BENCHMARK_SRC})                           
set_property(TARGET pytimeloop_bench PROPERTY CXX_STANDARD 17)              
target_include_directories(pytimeloop_bench PRIVATE                         
    ${CMAKE_SOURCE_DIR}/include                                                                 
    ${TIMELOOP_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/lib/pybind11/include)                                                   
target_link_libraries(pytimeloop_bench PRIVATE benchmark::benchmark pytimeloop define_gterminate ${LIBTIMELOOP} config++ yaml-cpp)
