# Dark Pool Detector - Machine Learning Library
add_library(darkpool_ml STATIC
    elastic_net.cpp
    feature_engineering.cpp
    inference_engine.cpp
    lstm_autoencoder.cpp
    transformer_model.cpp
)

target_include_directories(darkpool_ml PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Find ONNX Runtime


target_link_libraries(darkpool_ml PUBLIC
    darkpool_utils
    
    ${TBB_LIBRARIES}
    Threads::Threads
    m  # Math library
)

# ML-specific optimizations
target_compile_options(darkpool_ml PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
    -finline-functions
    -ftree-vectorize
    -mavx2
    -mavx512f
    -mfma
    -fopenmp  # OpenMP for parallel feature extraction
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

# SIMD optimizations
target_compile_definitions(darkpool_ml PRIVATE
    USE_AVX2=1
    USE_AVX512=1
)

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_ml PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Export for other modules
set_target_properties(darkpool_ml PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)


