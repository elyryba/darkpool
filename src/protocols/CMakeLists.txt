# High-performance protocol parsers
add_library(darkpool_protocols STATIC
    fix_parser.cpp
    itch_parser.cpp
    ouch_parser.cpp
    protocol_normalizer.cpp
)

target_include_directories(darkpool_protocols PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(darkpool_protocols PRIVATE
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
    target_compile_options(darkpool_protocols PRIVATE -finline-limit=1000)
endif()

target_compile_definitions(darkpool_protocols PRIVATE
    ZERO_COPY_PARSING=1
    BRANCH_PREDICTION_HINTS=1
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_protocols PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set_target_properties(darkpool_protocols PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
