# Dark Pool Detector - Utilities Library
add_library(darkpool_utils STATIC
    cpu_affinity.cpp
    lock_free_queue.cpp
    memory_pool.cpp
    metrics_collector.cpp
    ring_buffer.cpp
)

target_include_directories(darkpool_utils PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Find required libraries
find_package(prometheus-cpp REQUIRED)
find_package(NUMA REQUIRED)

target_link_libraries(darkpool_utils PUBLIC
    prometheus-cpp::core
    prometheus-cpp::push
    ${NUMA_LIBRARIES}
    Threads::Threads
)

# Lock-free utilities require extreme optimization
target_compile_options(darkpool_utils PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
    -finline-functions
    -finline-limit=2000  # Aggressive inlining for lock-free code
    -ftree-vectorize
    -mavx2
    -mfma
    -fno-exceptions  # No exceptions in utilities
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

# NUMA and cache optimizations
target_compile_definitions(darkpool_utils PRIVATE
    CACHE_LINE_SIZE=64
    NUMA_AWARE_ALLOCATION=1
    USE_HUGE_PAGES=1
)

# Platform-specific optimizations
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(darkpool_utils PRIVATE
        LINUX_PERF_EVENTS=1
    )
endif()

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_utils PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Export for other modules
set_target_properties(darkpool_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
