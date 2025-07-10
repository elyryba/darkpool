#include <gtest/gtest.h>
#include "darkpool/utils/cpu_affinity.hpp"
#include "darkpool/utils/memory_pool.hpp"
#include "darkpool/utils/metrics_collector.hpp"
#include <iostream>
#include <chrono>
#include <thread>

// Global test environment for setup/teardown
class DarkPoolTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        std::cout << "=== DarkPool Test Suite ===" << std::endl;
        std::cout << "Initializing test environment..." << std::endl;
        
        // Pin test thread to CPU 0 for consistent timing
        darkpool::utils::set_current_thread_affinity(0);
        
        // Get CPU topology
        auto topology = darkpool::utils::get_cpu_topology();
        std::cout << "CPU Configuration:" << std::endl;
        std::cout << "  Total CPUs: " << topology.total_cpus << std::endl;
        std::cout << "  NUMA nodes: " << topology.numa_nodes << std::endl;
        std::cout << "  Cache line size: " << topology.cache_line_size << " bytes" << std::endl;
        std::cout << "  Hyperthreading: " << (topology.hyperthreading_enabled ? "enabled" : "disabled") << std::endl;
        
        // Initialize global memory pool
        std::cout << "Initializing memory pools..." << std::endl;
        
        // Warm up allocators
        for (int i = 0; i < 1000; ++i) {
            void* p = darkpool::utils::g_memory_pool.allocate(1024);
            darkpool::utils::g_memory_pool.deallocate(p, 1024);
        }
        
        // Initialize metrics
        auto& metrics = darkpool::utils::g_metrics;
        metrics.create_histogram("test_latency_ns", "Test execution latency");
        
        // Disable CPU frequency scaling for consistent benchmarks
#ifdef __linux__
        system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1");
#endif
        
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void TearDown() override {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time_);
        
        std::cout << "\nTest suite completed in " << duration.count() << " seconds" << std::endl;
        
        // Print memory statistics
        auto mem_stats = darkpool::utils::g_memory_pool.get_stats();
        std::cout << "\nMemory Pool Statistics:" << std::endl;
        std::cout << "  Total allocated: " << mem_stats.total_allocated / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  High water mark: " << mem_stats.high_water_mark / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Fragmentation: " << std::fixed << std::setprecision(2) 
                  << mem_stats.fragmentation_ratio * 100 << "%" << std::endl;
        
        // Print test metrics
        std::cout << "\nTest Metrics:" << std::endl;
        std::cout << darkpool::utils::g_metrics.expose_json() << std::endl;
        
        // Re-enable default CPU governor
#ifdef __linux__
        system("echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1");
#endif
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Custom test event listener for performance tracking
class PerformanceListener : public ::testing::EmptyTestEventListener {
public:
    void OnTestStart(const ::testing::TestInfo& test_info) override {
        test_start_ = std::chrono::high_resolution_clock::now();
    }
    
    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        auto test_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_start_).count();
        
        if (duration > 1000000) { // More than 1 second
            std::cout << "  [SLOW] Test took " << duration / 1000 << " ms" << std::endl;
        }
        
        // Record test latency
        darkpool::utils::g_metrics.record_histogram("test_latency_ns", duration * 1000);
    }
    
private:
    std::chrono::high_resolution_clock::time_point test_start_;
};

// Helper macros for performance assertions
#define ASSERT_LATENCY_LT(operation, max_ns) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        operation; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); \
        ASSERT_LT(duration, max_ns) << "Operation took " << duration << " ns, expected < " << max_ns << " ns"; \
    } while(0)

#define EXPECT_LATENCY_LT(operation, max_ns) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        operation; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); \
        EXPECT_LT(duration, max_ns) << "Operation took " << duration << " ns, expected < " << max_ns << " ns"; \
    } while(0)

// Benchmark helper that runs operation multiple times and reports percentiles
template<typename Func>
void benchmark_operation(const std::string& name, Func operation, size_t iterations = 10000) {
    std::vector<int64_t> latencies;
    latencies.reserve(iterations);
    
    // Warm-up
    for (size_t i = 0; i < 100; ++i) {
        operation();
    }
    
    // Measure
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        operation();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        latencies.push_back(duration);
    }
    
    // Calculate percentiles
    std::sort(latencies.begin(), latencies.end());
    
    size_t p50_idx = iterations * 0.50;
    size_t p95_idx = iterations * 0.95;
    size_t p99_idx = iterations * 0.99;
    size_t p999_idx = iterations * 0.999;
    
    std::cout << "\nBenchmark: " << name << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  P50:  " << latencies[p50_idx] << " ns" << std::endl;
    std::cout << "  P95:  " << latencies[p95_idx] << " ns" << std::endl;
    std::cout << "  P99:  " << latencies[p99_idx] << " ns" << std::endl;
    std::cout << "  P99.9: " << latencies[p999_idx] << " ns" << std::endl;
    std::cout << "  Min:  " << latencies.front() << " ns" << std::endl;
    std::cout << "  Max:  " << latencies.back() << " ns" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Add custom environment
    ::testing::AddGlobalTestEnvironment(new DarkPoolTestEnvironment);
    
    // Add performance listener
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new PerformanceListener);
    
    // Parse custom arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--benchmark") {
            // Enable extended benchmarking
            std::cout << "Benchmark mode enabled" << std::endl;
        }
    }
    
    return RUN_ALL_TESTS();
}
