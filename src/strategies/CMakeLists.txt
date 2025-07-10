# Dark Pool Detector - Trading Strategies Library
add_library(darkpool_strategies STATIC
    cross_venue_optimizer.cpp
    dark_pool_strategy.cpp
    execution_optimizer.cpp
)

target_include_directories(darkpool_strategies PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(darkpool_strategies PUBLIC
    darkpool_utils
    darkpool_core
    ${TBB_LIBRARIES}
    Threads::Threads
    m  # Math library for optimization algorithms
)

# Strategy execution requires <1ms decision latency
target_compile_options(darkpool_strategies PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
    -finline-functions
    -ftree-vectorize
    -mavx2
    -mfma
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

# Strategy-specific optimizations
target_compile_definitions(darkpool_strategies PRIVATE
    ENABLE_ANTI_GAMING=1
    USE_SMART_ORDER_ROUTING=1
    OPTIMIZE_EXECUTION_ALGOS=1
)

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_strategies PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Export for other modules
set_target_properties(darkpool_strategies PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
