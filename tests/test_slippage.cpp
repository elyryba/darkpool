#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include "darkpool/core/slippage_tracker.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::core;

class SlippageTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tracker_ = std::make_unique<SlippageTracker>();
        
        // Pin to CPU for consistent timing
        utils::set_cpu_affinity(0);
        
        // Warm up
        volatile uint64_t dummy = 0;
        for (int i = 0; i < 1000000; ++i) {
            dummy += i;
        }
    }
    
    void TearDown() override {
        tracker_.reset();
    }
    
    // Create test execution
    Execution create_execution(const std::string& order_id,
                              const std::string& symbol,
                              Side side,
                              uint32_t quantity,
                              double expected_price,
                              double actual_price,
                              uint64_t latency_ns = 100000) {
        Execution exec;
        exec.order_id = order_id;
        exec.symbol = symbol;
        exec.side = side;
        exec.quantity = quantity;
        exec.expected_price = expected_price;
        exec.actual_price = actual_price;
        exec.timestamp = std::chrono::system_clock::now();
        exec.latency_ns = latency_ns;
        
        return exec;
    }
    
    // Create multi-fill execution sequence
    std::vector<Execution> create_multi_fill_sequence(const std::string& order_id,
                                                     const std::string& symbol,
                                                     Side side,
                                                     uint32_t total_quantity,
                                                     double expected_price,
                                                     const std::vector<double>& fill_prices) {
        std::vector<Execution> executions;
        uint32_t remaining = total_quantity;
        auto base_time = std::chrono::system_clock::now();
        
        for (size_t i = 0; i < fill_prices.size(); ++i) {
            uint32_t fill_qty = remaining / (fill_prices.size() - i);
            
            Execution exec;
            exec.order_id = order_id;
            exec.symbol = symbol;
            exec.side = side;
            exec.quantity = fill_qty;
            exec.expected_price = expected_price;
            exec.actual_price = fill_prices[i];
            exec.timestamp = base_time + std::chrono::milliseconds(i * 100);
            exec.latency_ns = 100000 + i * 10000; // Increasing latency
            
            executions.push_back(exec);
            remaining -= fill_qty;
        }
        
        return executions;
    }
    
    inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    std::unique_ptr<SlippageTracker> tracker_;
};

// Test basic slippage calculation
TEST_F(SlippageTrackerTest, BasicSlippageCalculation) {
    // Buy order with positive slippage (paid more than expected)
    auto exec = create_execution("ORDER1", "AAPL", Side::BUY, 100, 150.00, 150.05);
    
    auto result = tracker_->add_execution(exec);
    ASSERT_TRUE(result.has_value());
    
    // For buy: slippage = (actual - expected) / expected
    double expected_slippage = (150.05 - 150.00) / 150.00;
    EXPECT_NEAR(result->slippage_bps, expected_slippage * 10000, 0.01); // ~3.33 bps
    EXPECT_GT(result->slippage_bps, 0); // Positive slippage for buy
}

TEST_F(SlippageTrackerTest, SellSlippageCalculation) {
    // Sell order with negative slippage (sold for less than expected)
    auto exec = create_execution("ORDER2", "MSFT", Side::SELL, 200, 300.00, 299.90);
    
    auto result = tracker_->add_execution(exec);
    ASSERT_TRUE(result.has_value());
    
    // For sell: slippage = (expected - actual) / expected
    double expected_slippage = (300.00 - 299.90) / 300.00;
    EXPECT_NEAR(result->slippage_bps, expected_slippage * 10000, 0.01); // ~3.33 bps
    EXPECT_GT(result->slippage_bps, 0); // Positive slippage (unfavorable)
}

// Test market impact decomposition
TEST_F(SlippageTrackerTest, MarketImpactDecomposition) {
    // Large order with significant impact
    auto exec = create_execution("LARGE1", "GOOGL", Side::BUY, 10000, 2750.00, 2752.75);
    
    auto result = tracker_->add_execution(exec);
    ASSERT_TRUE(result.has_value());
    
    // Verify impact decomposition
    EXPECT_GT(result->permanent_impact_bps, 0);
    EXPECT_GT(result->temporary_impact_bps, 0);
    EXPECT_NEAR(result->slippage_bps, 
                result->permanent_impact_bps + result->temporary_impact_bps, 0.1);
    
    // Large orders should have higher temporary impact
    EXPECT_GT(result->temporary_impact_bps, result->permanent_impact_bps);
}

// Test multiple fills tracking
TEST_F(SlippageTrackerTest, MultipleFillsTracking) {
    std::vector<double> fill_prices = {150.00, 150.02, 150.05, 150.08};
    auto executions = create_multi_fill_sequence("MULTI1", "AAPL", Side::BUY, 
                                                1000, 150.00, fill_prices);
    
    SlippageResult final_result;
    
    for (const auto& exec : executions) {
        auto result = tracker_->add_execution(exec);
        ASSERT_TRUE(result.has_value());
        final_result = *result;
    }
    
    // Check aggregated slippage
    auto order_stats = tracker_->get_order_statistics("MULTI1");
    ASSERT_TRUE(order_stats.has_value());
    
    EXPECT_EQ(order_stats->fill_count, 4);
    EXPECT_EQ(order_stats->total_quantity, 1000);
    
    // VWAP should be between min and max fill prices
    EXPECT_GT(order_stats->vwap, 150.00);
    EXPECT_LT(order_stats->vwap, 150.08);
    
    // Total slippage should reflect average fill price vs expected
    double expected_vwap = (150.00 + 150.02 + 150.05 + 150.08) / 4;
    EXPECT_NEAR(order_stats->vwap, expected_vwap, 0.01);
}

// Test large order handling
TEST_F(SlippageTrackerTest, LargeOrderHandling) {
    // Test orders of different sizes
    struct TestCase {
        uint32_t quantity;
        double price_impact;
        const char* description;
    };
    
    TestCase cases[] = {
        {100, 0.01, "Small order"},
        {1000, 0.05, "Medium order"},
        {10000, 0.20, "Large order"},
        {100000, 1.00, "Very large order"}
    };
    
    for (const auto& tc : cases) {
        auto exec = create_execution(
            std::string("ORDER_") + tc.description,
            "SPY", Side::BUY, tc.quantity, 
            440.00, 440.00 * (1 + tc.price_impact / 100)
        );
        
        auto result = tracker_->add_execution(exec);
        ASSERT_TRUE(result.has_value()) << "Failed for " << tc.description;
        
        // Larger orders should have higher impact
        if (tc.quantity >= 10000) {
            EXPECT_GT(result->slippage_bps, 10.0) << "Large order should have significant slippage";
            EXPECT_TRUE(result->is_anomaly) << "Large order should be flagged as anomaly";
        }
    }
}

// Test performance metrics
TEST_F(SlippageTrackerTest, PerformanceMetrics) {
    auto exec = create_execution("PERF1", "AAPL", Side::BUY, 100, 150.00, 150.05);
    
    // Warm up
    for (int i = 0; i < 10000; ++i) {
        tracker_->add_execution(exec);
    }
    
    // Clear state
    tracker_->reset();
    
    // Measure
    const int iterations = 100000;
    std::vector<uint64_t> cycles;
    cycles.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        exec.order_id = "ORDER" + std::to_string(i);
        
        uint64_t start = rdtsc();
        auto result = tracker_->add_execution(exec);
        uint64_t end = rdtsc();
        
        cycles.push_back(end - start);
    }
    
    // Calculate percentiles
    std::sort(cycles.begin(), cycles.end());
    uint64_t p50 = cycles[iterations * 0.50];
    uint64_t p95 = cycles[iterations * 0.95];
    uint64_t p99 = cycles[iterations * 0.99];
    
    const double cycles_per_ns = 3.0;
    
    std::cout << "Slippage Tracker Performance:\n"
              << "  P50: " << p50 / cycles_per_ns << " ns\n"
              << "  P95: " << p95 / cycles_per_ns << " ns\n"
              << "  P99: " << p99 / cycles_per_ns << " ns\n";
    
    // Should process in <1μs
    EXPECT_LT(p99 / cycles_per_ns, 1000.0);
}

// Test edge cases
TEST_F(SlippageTrackerTest, EdgeCases) {
    // Zero volume
    auto zero_vol = create_execution("ZERO1", "AAPL", Side::BUY, 0, 150.00, 150.00);
    auto result = tracker_->add_execution(zero_vol);
    EXPECT_FALSE(result.has_value());
    
    // Extreme prices
    auto extreme_price = create_execution("EXTREME1", "PENNY", Side::BUY, 
                                         1000000, 0.0001, 0.0005);
    result = tracker_->add_execution(extreme_price);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->slippage_bps, 10000); // 400% slippage!
    EXPECT_TRUE(result->is_anomaly);
    
    // Same expected and actual price
    auto no_slip = create_execution("NOSLIP1", "AAPL", Side::BUY, 100, 150.00, 150.00);
    result = tracker_->add_execution(no_slip);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->slippage_bps, 0.0, 0.01);
    
    // Negative prices (should be rejected)
    auto negative_price = create_execution("NEG1", "ERROR", Side::BUY, 100, -10.00, 150.00);
    result = tracker_->add_execution(negative_price);
    EXPECT_FALSE(result.has_value());
}

// Test symbol-level statistics
TEST_F(SlippageTrackerTest, SymbolLevelStatistics) {
    // Add executions for multiple symbols
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL"};
    
    for (const auto& symbol : symbols) {
        for (int i = 0; i < 100; ++i) {
            double price = 100.0 + std::hash<std::string>{}(symbol) % 100;
            double slippage = (i % 10) * 0.001; // 0-0.9% slippage
            
            auto exec = create_execution(
                symbol + "_" + std::to_string(i),
                symbol, 
                i % 2 ? Side::BUY : Side::SELL,
                100 + i,
                price,
                price * (1 + slippage)
            );
            
            tracker_->add_execution(exec);
        }
    }
    
    // Check symbol statistics
    for (const auto& symbol : symbols) {
        auto stats = tracker_->get_symbol_statistics(symbol);
        ASSERT_TRUE(stats.has_value());
        
        EXPECT_EQ(stats->total_executions, 100);
        EXPECT_GT(stats->avg_slippage_bps, 0);
        EXPECT_GT(stats->total_volume, 10000);
        
        // Should have both buy and sell executions
        EXPECT_GT(stats->buy_executions, 0);
        EXPECT_GT(stats->sell_executions, 0);
    }
}

// Test anomaly detection
TEST_F(SlippageTrackerTest, AnomalyDetection) {
    // Normal execution
    auto normal = create_execution("NORMAL1", "AAPL", Side::BUY, 100, 150.00, 150.01);
    auto result = tracker_->add_execution(normal);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->is_anomaly);
    
    // High slippage execution
    auto high_slip = create_execution("ANOMALY1", "AAPL", Side::BUY, 100, 150.00, 151.50);
    result = tracker_->add_execution(high_slip);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->is_anomaly);
    EXPECT_EQ(result->anomaly_type, AnomalyType::HIGH_SLIPPAGE);
    
    // High latency execution
    auto high_latency = create_execution("LATENCY1", "AAPL", Side::BUY, 
                                        100, 150.00, 150.10, 10000000); // 10ms
    result = tracker_->add_execution(high_latency);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->is_anomaly);
    EXPECT_TRUE(result->anomaly_reasons.find("high_latency") != result->anomaly_reasons.end());
}

// Test information leakage detection
TEST_F(SlippageTrackerTest, InformationLeakageDetection) {
    // Simulate pattern of increasing slippage (potential info leakage)
    std::string order_id = "LEAK1";
    
    for (int i = 0; i < 10; ++i) {
        double slippage_pct = 0.001 * (i + 1); // Increasing slippage
        auto exec = create_execution(
            order_id + "_" + std::to_string(i),
            "AAPL", Side::BUY, 1000, 150.00, 
            150.00 * (1 + slippage_pct)
        );
        
        auto result = tracker_->add_execution(exec);
        ASSERT_TRUE(result.has_value());
        
        if (i >= 5) {
            // Should detect pattern after several executions
            EXPECT_TRUE(result->information_leakage_suspected);
        }
    }
}

// Test reset functionality
TEST_F(SlippageTrackerTest, ResetFunctionality) {
    // Add some executions
    for (int i = 0; i < 100; ++i) {
        auto exec = create_execution("ORDER" + std::to_string(i), 
                                    "AAPL", Side::BUY, 100, 150.00, 150.01);
        tracker_->add_execution(exec);
    }
    
    // Verify data exists
    auto stats = tracker_->get_symbol_statistics("AAPL");
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->total_executions, 100);
    
    // Reset
    tracker_->reset();
    
    // Verify data cleared
    stats = tracker_->get_symbol_statistics("AAPL");
    EXPECT_FALSE(stats.has_value());
}

// Test concurrent access
TEST_F(SlippageTrackerTest, ConcurrentAccess) {
    const int num_threads = 4;
    const int execs_per_thread = 10000;
    
    std::vector<std::thread> threads;
    std::atomic<int> total_processed{0};
    
    auto worker = [&](int thread_id) {
        utils::set_cpu_affinity(thread_id);
        
        for (int i = 0; i < execs_per_thread; ++i) {
            auto exec = create_execution(
                "T" + std::to_string(thread_id) + "_" + std::to_string(i),
                "SYMBOL" + std::to_string(thread_id),
                Side::BUY, 100, 100.00, 100.00 + (i % 10) * 0.01
            );
            
            if (tracker_->add_execution(exec).has_value()) {
                total_processed++;
            }
        }
    };
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(total_processed.load(), num_threads * execs_per_thread);
}

// Benchmark different scenarios
TEST_F(SlippageTrackerTest, BenchmarkScenarios) {
    struct Scenario {
        const char* name;
        uint32_t quantity;
        double slippage_pct;
        int fills;
    };
    
    Scenario scenarios[] = {
        {"Small single fill", 100, 0.001, 1},
        {"Large single fill", 10000, 0.005, 1},
        {"Small multi-fill", 100, 0.002, 5},
        {"Large multi-fill", 10000, 0.008, 10}
    };
    
    for (const auto& scenario : scenarios) {
        tracker_->reset();
        
        // Create executions
        std::vector<Execution> execs;
        if (scenario.fills == 1) {
            execs.push_back(create_execution("BENCH1", "TEST", Side::BUY,
                                           scenario.quantity, 100.00,
                                           100.00 * (1 + scenario.slippage_pct)));
        } else {
            for (int i = 0; i < scenario.fills; ++i) {
                execs.push_back(create_execution("BENCH1", "TEST", Side::BUY,
                                               scenario.quantity / scenario.fills,
                                               100.00,
                                               100.00 * (1 + scenario.slippage_pct * (i + 1) / scenario.fills)));
            }
        }
        
        // Warm up
        for (int i = 0; i < 1000; ++i) {
            for (const auto& exec : execs) {
                tracker_->add_execution(exec);
            }
            tracker_->reset();
        }
        
        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; ++i) {
            for (const auto& exec : execs) {
                tracker_->add_execution(exec);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double avg_ns = duration.count() / (10000.0 * execs.size());
        
        std::cout << "Slippage Tracker - " << scenario.name 
                  << ": " << avg_ns << " ns per execution\n";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
