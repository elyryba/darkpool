# Dark Pool Detector - Protocol Parsing Library
add_library(darkpool_protocols STATIC
    fix_parser.cpp
    itch_parser.cpp
    ouch_parser.cpp
    protocol_normalizer.cpp
)

target_include_directories(darkpool_protocols PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(darkpool_protocols PUBLIC
    darkpool_utils
    Threads::Threads
)

# Protocol parsing requires extreme optimization for <50ns performance
target_compile_options(darkpool_protocols PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
    -finline-functions
    -finline-limit=1000
    -ftree-vectorize
    -mavx2
    -mfma
    -fno-exceptions  # No exceptions in hot path
    -fno-rtti       # No RTTI overhead
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

# Zero-copy optimizations
target_compile_definitions(darkpool_protocols PRIVATE
    ZERO_COPY_PARSING=1
    BRANCH_PREDICTION_HINTS=1
)

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_protocols PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Export for other modules
set_target_properties(darkpool_protocols PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
