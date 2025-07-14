# Dark Pool Detector - Benchmarks
find_package(benchmark REQUIRED)

# Latency benchmark
add_executable(benchmark_latency
    benchmark_latency.cpp
)

target_include_directories(benchmark_latency PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(benchmark_latency PRIVATE
    darkpool_core
    darkpool_protocols
    darkpool_utils
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
)

# Throughput benchmark
add_executable(benchmark_throughput
    benchmark_throughput.cpp
)

target_include_directories(benchmark_throughput PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(benchmark_throughput PRIVATE
    darkpool_core
    darkpool_protocols
    darkpool_utils
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
)

# ML inference benchmark
add_executable(benchmark_ml_inference
    benchmark_ml_inference.cpp
)

target_include_directories(benchmark_ml_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(benchmark_ml_inference PRIVATE
    darkpool_ml
    darkpool_utils
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
)

# Protocol parsing benchmark
add_executable(benchmark_protocol_parsing
    benchmark_protocol_parsing.cpp
)

target_include_directories(benchmark_protocol_parsing PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(benchmark_protocol_parsing PRIVATE
    darkpool_protocols
    darkpool_utils
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
)

# Set properties for all benchmarks
set(BENCHMARK_TARGETS
    benchmark_latency
    benchmark_throughput
    benchmark_ml_inference
    benchmark_protocol_parsing
)

foreach(target ${BENCHMARK_TARGETS})
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    
    # Enable CPU-specific optimizations for benchmarks
    target_compile_options(${target} PRIVATE
        -O3
        -march=native
        -mtune=native
        -mavx2
        -mfma
    )
endforeach()

# Add benchmark tests for CTest
add_test(NAME benchmark_latency_test
         COMMAND benchmark_latency --benchmark_min_time=0.1)
add_test(NAME benchmark_throughput_test
         COMMAND benchmark_throughput --benchmark_min_time=0.1)
add_test(NAME benchmark_ml_inference_test
         COMMAND benchmark_ml_inference --benchmark_min_time=0.1)
add_test(NAME benchmark_protocol_parsing_test
         COMMAND benchmark_protocol_parsing --benchmark_min_time=0.1)

