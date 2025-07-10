# Dark Pool Detector - Core Detection Library
add_library(darkpool_core STATIC
    detector_impl.cpp
    execution_heatmap.cpp
    hawkes_process.cpp
    hidden_markov_model.cpp
    hidden_refill_detector.cpp
    order_book_imbalance.cpp
    pin_model.cpp
    post_trade_drift.cpp
    realtime_stream.cpp
    slippage_tracker.cpp
    trade_clustering.cpp
    trade_to_quote_ratio.cpp
    vpin_calculator.cpp
)

target_include_directories(darkpool_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(darkpool_core PUBLIC
    darkpool_utils
    darkpool_protocols
    ${TBB_LIBRARIES}
    Threads::Threads
    m  # Math library
)

# Optimization flags for Jane Street level performance
target_compile_options(darkpool_core PRIVATE
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

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Export for other modules
set_target_properties(darkpool_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
