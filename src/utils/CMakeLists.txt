# Low-level utilities and data structures
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

target_link_libraries(darkpool_utils PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_options(darkpool_utils PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
    -finline-functions
    -ftree-vectorize
    -mavx2
    -mfma
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(darkpool_utils PRIVATE -finline-limit=2000)
endif()

target_compile_definitions(darkpool_utils PRIVATE
    CACHE_LINE_SIZE=64
    NUMA_AWARE_ALLOCATION=1
    USE_HUGE_PAGES=1
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(darkpool_utils PRIVATE
        LINUX_PERF_EVENTS=1
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_utils PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set_target_properties(darkpool_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
