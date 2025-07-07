cmake_minimum_required(VERSION 3.20)
project(darkpool-detector VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ENABLE_TESTS "Enable unit tests" ON)
option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_SANITIZERS "Enable address and thread sanitizers" OFF)
option(ENABLE_NATIVE_ARCH "Enable -march=native" OFF)
option(ENABLE_LTO "Enable link-time optimization" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_MKL "Enable Intel MKL" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -fno-finite-math-only")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    if(ENABLE_NATIVE_ARCH)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
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
find_package(Boost 1.74 REQUIRED COMPONENTS system filesystem)
find_package(yaml-cpp REQUIRED)
find_package(Protobuf REQUIRED)
find_package(prometheus-cpp REQUIRED)

# Find optional packages
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(ENABLE_MKL)
    find_package(MKL REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(DARKPOOL_SOURCES
    src/main.cpp
    src/config.cpp
    
    # Core
    src/core/detector_impl.cpp
    src/core/execution_heatmap.cpp
    src/core/hawkes_process.cpp
    src/core/hidden_markov_model.cpp
    src/core/hidden_refill_detector.cpp
    src/core/order_book_imbalance.cpp
    src/core/pin_model.cpp
    src/core/post_trade_drift.cpp
    src/core/realtime_stream.cpp
    src/core/slippage_tracker.cpp
    src/core/trade_clustering.cpp
    src/core/trade_to_quote_ratio.cpp
    src/core/vpin_calculator.cpp
    
    # ML
    src/ml/elastic_net.cpp
    src/ml/feature_engineering.cpp
    src/ml/inference_engine.cpp
    src/ml/lstm_autoencoder.cpp
    src/ml/transformer_model.cpp
    
    # Protocols
    src/protocols/fix_parser.cpp
    src/protocols/itch_parser.cpp
    src/protocols/ouch_parser.cpp
    src/protocols/protocol_normalizer.cpp
    
    # Strategies
    src/strategies/cross_venue_optimizer.cpp
    src/strategies/dark_pool_strategy.cpp
    src/strategies/execution_optimizer.cpp
    
    # Utils
    src/utils/cpu_affinity.cpp
    src/utils/lock_free_queue.cpp
    src/utils/memory_pool.cpp
    src/utils/metrics_collector.cpp
    src/utils/ring_buffer.cpp
    
    # Visualization
    src/visualization/websocket_server.cpp
)

# Main executable
add_executable(darkpool_detector ${DARKPOOL_SOURCES})

# Link libraries
target_link_libraries(darkpool_detector
    PRIVATE
        Threads::Threads
        Boost::system
        Boost::filesystem
        yaml-cpp
        ${Protobuf_LIBRARIES}
        prometheus-cpp::core
        prometheus-cpp::push
        numa
        tbb
        zmq
        ssl
        crypto
        websocketpp
)

if(ENABLE_CUDA)
    target_link_libraries(darkpool_detector PRIVATE CUDA::cudart CUDA::cublas)
endif()

if(ENABLE_MKL)
    target_link_libraries(darkpool_detector PRIVATE MKL::MKL)
endif()

# Tests
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install
install(TARGETS darkpool_detector DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION etc/darkpool)
install(DIRECTORY visualization/ DESTINATION share/darkpool/visualization)
