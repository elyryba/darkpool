# Dark Pool Detector - Visualization Library
add_library(darkpool_visualization STATIC
    websocket_server.cpp
)

target_include_directories(darkpool_visualization PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Find WebSocket++ and Boost
find_package(websocketpp REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS system thread)

target_link_libraries(darkpool_visualization PUBLIC
    darkpool_utils
    darkpool_core
    websocketpp::websocketpp
    Boost::system
    Boost::thread
    Threads::Threads
)

# WebSocket server optimizations
target_compile_options(darkpool_visualization PRIVATE
    -O3
    -march=native
    -mtune=native
    -funroll-loops
    -finline-functions
    -ftree-vectorize
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

# WebSocket configuration
target_compile_definitions(darkpool_visualization PRIVATE
    ASIO_STANDALONE  # Use standalone ASIO
    _WEBSOCKETPP_CPP11_STL_
    _WEBSOCKETPP_CPP11_FUNCTIONAL_
)

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET darkpool_visualization PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Export for other modules
set_target_properties(darkpool_visualization PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
