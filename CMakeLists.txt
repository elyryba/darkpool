cmake_minimum_required(VERSION 3.20)
project(darkpool-detector VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTING "Enable unit tests" ON)
option(BUILD_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_SANITIZERS "Enable address and thread sanitizers" OFF)
option(ENABLE_NATIVE_ARCH "Enable -march=native" ON)
option(ENABLE_LTO "Enable link-time optimization" ON)
option(ENABLE_ONNX "Enable ONNX Runtime support" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops -finline-functions")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    if(ENABLE_NATIVE_ARCH)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -mavx2 -mfma")
    endif()
    if(ENABLE_LTO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3")
    if(ENABLE_SANITIZERS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
    endif()
endif()

# Find packages
find_package(Threads REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS system filesystem thread)
find_package(yaml-cpp REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)

# Find TBB
find_package(TBB REQUIRED)

# Find NUMA
find_package(PkgConfig REQUIRED)
pkg_check_modules(NUMA REQUIRED numa)

# Find ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED libzmq)

# Find WebSocketPP
find_path(WEBSOCKETPP_INCLUDE_DIR websocketpp/config.hpp)

# Find Google Test (for tests)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
endif()

# Find Google Benchmark (for benchmarks)
if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
endif()

# Find ONNX Runtime
if(ENABLE_ONNX)
    find_package(onnxruntime REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${WEBSOCKETPP_INCLUDE_DIR})

# Build subdirectories in dependency order
add_subdirectory(src/utils)        # No dependencies
add_subdirectory(src/protocols)    # Depends on utils
add_subdirectory(src/core)         # Depends on utils, protocols  
add_subdirectory(src/ml)           # Depends on utils
add_subdirectory(src/strategies)   # Depends on utils, core
add_subdirectory(src/visualization) # Depends on utils, core

# Main executable
add_executable(darkpool_detector src/main.cpp src/config.cpp)

# Link libraries
target_link_libraries(darkpool_detector
    PRIVATE
        darkpool_core
        darkpool_ml
        darkpool_protocols
        darkpool_strategies
        darkpool_visualization
        darkpool_utils
        yaml-cpp
        ${ZeroMQ_LIBRARIES}
        Threads::Threads
)

# Set properties
set_target_properties(darkpool_detector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Tests
if(BUILD_TESTING)
    enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install
install(TARGETS darkpool_detector DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION etc/darkpool)

# Package configuration
include(CPack)
set(CPACK_PACKAGE_NAME "darkpool-detector")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Jane Street-level Dark Pool Detection System")
set(CPACK_PACKAGE_VENDOR "DarkPool Technologies")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/darkpool")
