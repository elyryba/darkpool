# Dark Pool Detector - Tests
include(GoogleTest)

# Find Google Test
find_package(GTest REQUIRED)

# Test executable
add_executable(darkpool_tests
    test_main.cpp
    test_hawkes_process.cpp
    test_fix_parser.cpp
    test_itch_parser.cpp
    test_ml_inference.cpp
    test_order_book_imbalance.cpp
    test_slippage.cpp
    test_strategies.cpp
    test_tqr.cpp
)

target_include_directories(darkpool_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(darkpool_tests PRIVATE
    darkpool_core
    darkpool_ml
    darkpool_protocols
    darkpool_strategies
    darkpool_utils
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

# Set test properties
set_target_properties(darkpool_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Data generator tool
add_executable(data_generator
    data_generator.cpp
)

target_include_directories(data_generator PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(data_generator PRIVATE
    darkpool_utils
    darkpool_protocols
    Threads::Threads
)

set_target_properties(data_generator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Enable test discovery
gtest_discover_tests(darkpool_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROPERTIES TIMEOUT 120
)

# Add individual test targets for CI
add_test(NAME test_protocols 
         COMMAND darkpool_tests --gtest_filter="*Parser*:*Protocol*")
add_test(NAME test_core 
         COMMAND darkpool_tests --gtest_filter="*Hawkes*:*TQR*:*Slippage*:*OrderBook*")
add_test(NAME test_ml 
         COMMAND darkpool_tests --gtest_filter="*ML*:*Inference*")
add_test(NAME test_strategies 
         COMMAND darkpool_tests --gtest_filter="*Strategy*:*Strategies*")

# Performance validation tests
add_test(NAME validate_fix_parsing_performance
         COMMAND darkpool_tests --gtest_filter="FIXParserTest.PerformanceSingleMessage")
set_tests_properties(validate_fix_parsing_performance PROPERTIES
    PASS_REGULAR_EXPRESSION "FIX Parser Performance.*P99.*[0-9]+\\.[0-9]+ ns"
    FAIL_REGULAR_EXPRESSION "exceeds.*100.*ns"
)

add_test(NAME validate_itch_parsing_performance
         COMMAND darkpool_tests --gtest_filter="ITCHParserTest.PerformanceSingleMessage")
set_tests_properties(validate_itch_parsing_performance PROPERTIES
    PASS_REGULAR_EXPRESSION "ITCH Parser Performance.*P99.*[0-9]+\\.[0-9]+ ns"
    FAIL_REGULAR_EXPRESSION "exceeds.*50.*ns"
)

add_test(NAME validate_ml_inference_performance
         COMMAND darkpool_tests --gtest_filter="MLInferenceTest.PerformanceBatchInference")
set_tests_properties(validate_ml_inference_performance PROPERTIES
    PASS_REGULAR_EXPRESSION "SUCCESS.*P99.*latency.*ms.*meets.*2\\.3ms"
)

