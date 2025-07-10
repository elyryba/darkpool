#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include "darkpool/core/order_book_imbalance.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::core;

class OrderBookImbalanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        detector_ = std::make_unique<OrderBookImbalanceDetector>();
        
        // Pin to CPU for consistent timing
        utils::set_cpu_affinity(0);
        
        // Warm up
        volatile uint64_t dummy = 0;
        for (int i = 0; i < 1000000; ++i) {
            dummy += i;
        }
    }
    
    void TearDown() override {
        detector_.reset();
    }
    
    // Create realistic order book update
    OrderBookUpdate create_book_update(const std::string& symbol,
                                     const std::vector<std::pair<double, uint32_t>>& bids,
                                     const std::vector<std::pair<double, uint32_t>>& asks) {
        OrderBookUpdate update;
        update.symbol = symbol;
        update.timestamp = std::chrono::system_clock::now();
        update.sequence_number = sequence_++;
        
        // Add bid levels
        for (size_t i = 0; i < bids.size() && i < 10; ++i) {
            update.bid_levels[i] = {bids[i].first, bids[i].second};
        }
        update.bid_depth = std::min(bids.size(), size_t(10));
        
        // Add ask levels
        for (size_t i = 0; i < asks.size() && i < 10; ++i) {
            update.ask_levels[i] = {asks[i].first, asks[i].second};
        }
        update.ask_depth = std::min(asks.size(), size_t(10));
        
        return update;
    }
    
    // Create balanced book
    OrderBookUpdate create_balanced_book(const std::string& symbol, double mid_price) {
        std::vector<std::pair<double, uint32_t>> bids, asks;
        
        // Create 10 levels each side
        for (int i = 0; i < 10; ++i) {
            double bid_price = mid_price - 0.01 * (i + 1);
            double ask_price = mid_price + 0.01 * (i + 1);
            uint32_t size = 1000 * (i + 1); // Increasing size with depth
            
            bids.push_back({bid_price, size});
            asks.push_back({ask_price, size});
        }
        
        return create_book_update(symbol, bids, asks);
    }
    
    // Create imbalanced book
    OrderBookUpdate create_imbalanced_book(const std::string& symbol, double mid_price, 
                                         double imbalance_ratio) {
        std::vector<std::pair<double, uint32_t>> bids, asks;
        
        for (int i = 0; i < 10; ++i) {
            double bid_price = mid_price - 0.01 * (i + 1);
            double ask_price = mid_price + 0.01 * (i + 1);
            
            uint32_t bid_size = static_cast<uint32_t>(1000 * (i + 1) * imbalance_ratio);
            uint32_t ask_size = 1000 * (i + 1);
            
            bids.push_back({bid_price, bid_size});
            asks.push_back({ask_price, ask_size});
        }
        
        return create_book_update(symbol, bids, asks);
    }
    
    inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    std::unique_ptr<OrderBookImbalanceDetector> detector_;
    uint64_t sequence_ = 1;
};

// Test basic imbalance calculation
TEST_F(OrderBookImbalanceTest, BasicImbalanceCalculation) {
    auto update = create_balanced_book("AAPL", 150.50);
    
    auto result = detector_->process_update(update);
    ASSERT_TRUE(result.has_value());
    
    // Balanced book should have near-zero imbalance
    EXPECT_NEAR(result->imbalance_ratio, 0.0, 0.01);
    EXPECT_FALSE(result->is_anomaly);
}

TEST_F(OrderBookImbalanceTest, HeavyBidImbalance) {
    auto update = create_imbalanced_book("AAPL", 150.50, 3.0); // 3x bid size
    
    auto result = detector_->process_update(update);
    ASSERT_TRUE(result.has_value());
    
    // Should detect bid-side imbalance
    EXPECT_GT(result->imbalance_ratio, 0.4);
    EXPECT_TRUE(result->is_anomaly);
    EXPECT_EQ(result->anomaly_type, AnomalyType::BID_PRESSURE);
}

TEST_F(OrderBookImbalanceTest, HeavyAskImbalance) {
    auto update = create_imbalanced_book("AAPL", 150.50, 0.33); // 3x ask size
    
    auto result = detector_->process_update(update);
    ASSERT_TRUE(result.has_value());
    
    // Should detect ask-side imbalance
    EXPECT_LT(result->imbalance_ratio, -0.4);
    EXPECT_TRUE(result->is_anomaly);
    EXPECT_EQ(result->anomaly_type, AnomalyType::ASK_PRESSURE);
}

// Test multi-level imbalance
TEST_F(OrderBookImbalanceTest, MultiLevelImbalance) {
    std::vector<std::pair<double, uint32_t>> bids = {
        {150.49, 5000},  // Heavy at top
        {150.48, 4000},
        {150.47, 3000},
        {150.46, 1000},
        {150.45, 1000}
    };
    
    std::vector<std::pair<double, uint32_t>> asks = {
        {150.51, 500},   // Light at top
        {150.52, 600},
        {150.53, 700},
        {150.54, 800},
        {150.55, 900}
    };
    
    auto update = create_book_update("AAPL", bids, asks);
    auto result = detector_->process_update(update);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->top_level_imbalance, 0.8); // Strong top-level imbalance
    EXPECT_GT(result->weighted_imbalance, 0.5);  // Overall bid pressure
}

// Test quote-based book building
TEST_F(OrderBookImbalanceTest, QuoteBasedBookBuilding) {
    Quote quote;
    quote.symbol = "MSFT";
    quote.bid_price = 350.00;
    quote.bid_size = 2000;
    quote.ask_price = 350.02;
    quote.ask_size = 1500;
    quote.timestamp = std::chrono::system_clock::now();
    
    // Process multiple quotes to build book
    for (int i = 0; i < 100; ++i) {
        detector_->process_quote(quote);
        
        // Vary quote slightly
        quote.bid_price -= 0.01;
        quote.ask_price += 0.01;
        quote.bid_size += 100;
        quote.ask_size -= 50;
    }
    
    // Get current imbalance
    auto imbalance = detector_->get_current_imbalance("MSFT");
    ASSERT_TRUE(imbalance.has_value());
    EXPECT_GT(imbalance->imbalance_ratio, 0.0); // Should show bid pressure
}

// Test performance with deep books
TEST_F(OrderBookImbalanceTest, PerformanceDeepBook) {
    // Create deep 10-level book
    auto update = create_balanced_book("AAPL", 150.50);
    
    // Warm up
    for (int i = 0; i < 10000; ++i) {
        detector_->process_update(update);
    }
    
    // Measure
    const int iterations = 100000;
    std::vector<uint64_t> cycles;
    cycles.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        // Modify update slightly each time
        update.bid_levels[0].size += (i % 10);
        update.sequence_number = i;
        
        uint64_t start = rdtsc();
        auto result = detector_->process_update(update);
        uint64_t end = rdtsc();
        
        cycles.push_back(end - start);
    }
    
    // Calculate percentiles
    std::sort(cycles.begin(), cycles.end());
    uint64_t p50 = cycles[iterations * 0.50];
    uint64_t p95 = cycles[iterations * 0.95];
    uint64_t p99 = cycles[iterations * 0.99];
    
    const double cycles_per_ns = 3.0;
    
    std::cout << "Order Book Imbalance Performance (10-level book):\n"
              << "  P50: " << p50 / cycles_per_ns << " ns\n"
              << "  P95: " << p95 / cycles_per_ns << " ns\n"
              << "  P99: " << p99 / cycles_per_ns << " ns\n";
    
    // Should process in <500ns even with 10 levels
    EXPECT_LT(p99 / cycles_per_ns, 500.0);
}

// Test memory efficiency
TEST_F(OrderBookImbalanceTest, MemoryEfficiency) {
    // Track allocations
    size_t initial_memory = detector_->get_memory_usage();
    
    // Process many updates for multiple symbols
    const std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"};
    
    for (int i = 0; i < 100000; ++i) {
        for (const auto& symbol : symbols) {
            auto update = create_balanced_book(symbol, 100.0 + i * 0.01);
            detector_->process_update(update);
        }
    }
    
    size_t final_memory = detector_->get_memory_usage();
    size_t growth = final_memory - initial_memory;
    
    // Memory growth should be minimal (only for book state per symbol)
    size_t expected_max = symbols.size() * sizeof(OrderBook) * 2; // Double buffer
    EXPECT_LT(growth, expected_max);
}

// Test anomaly detection thresholds
TEST_F(OrderBookImbalanceTest, AnomalyThresholds) {
    struct TestCase {
        double imbalance_ratio;
        bool should_detect;
        const char* description;
    };
    
    TestCase cases[] = {
        {1.0, false, "Perfectly balanced"},
        {1.5, false, "Slight bid pressure"},
        {2.0, true, "Moderate bid pressure"},
        {3.0, true, "Heavy bid pressure"},
        {0.5, true, "Moderate ask pressure"},
        {0.33, true, "Heavy ask pressure"}
    };
    
    for (const auto& tc : cases) {
        auto update = create_imbalanced_book("TEST", 100.0, tc.imbalance_ratio);
        auto result = detector_->process_update(update);
        
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->is_anomaly, tc.should_detect) 
            << "Failed for " << tc.description 
            << " (ratio: " << tc.imbalance_ratio << ")";
    }
}

// Test book reset handling
TEST_F(OrderBookImbalanceTest, BookResetHandling) {
    // Build up book state
    for (int i = 0; i < 10; ++i) {
        auto update = create_balanced_book("AAPL", 150.0 + i * 0.1);
        detector_->process_update(update);
    }
    
    // Send reset signal
    OrderBookUpdate reset;
    reset.symbol = "AAPL";
    reset.timestamp = std::chrono::system_clock::now();
    reset.is_snapshot = true;
    reset.bid_depth = 0;
    reset.ask_depth = 0;
    
    auto result = detector_->process_update(reset);
    ASSERT_TRUE(result.has_value());
    
    // Verify book was reset
    auto imbalance = detector_->get_current_imbalance("AAPL");
    EXPECT_FALSE(imbalance.has_value()); // No imbalance after reset
}

// Test hidden liquidity detection
TEST_F(OrderBookImbalanceTest, HiddenLiquidityDetection) {
    // Create book with suspiciously thin top levels
    std::vector<std::pair<double, uint32_t>> bids = {
        {150.49, 100},    // Very thin top
        {150.48, 100},    // Very thin
        {150.47, 5000},   // Heavy deeper
        {150.46, 6000},
        {150.45, 7000}
    };
    
    std::vector<std::pair<double, uint32_t>> asks = {
        {150.51, 100},    // Very thin top
        {150.52, 100},    // Very thin
        {150.53, 5000},   // Heavy deeper
        {150.54, 6000},
        {150.55, 7000}
    };
    
    auto update = create_book_update("AAPL", bids, asks);
    auto result = detector_->process_update(update);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->hidden_liquidity_suspected);
    EXPECT_GT(result->depth_imbalance_ratio, 10.0); // Deep vs top imbalance
}

// Test rapid changes detection
TEST_F(OrderBookImbalanceTest, RapidChangesDetection) {
    // Start with balanced book
    auto update1 = create_balanced_book("AAPL", 150.50);
    detector_->process_update(update1);
    
    // Rapid flip to heavy bid
    auto update2 = create_imbalanced_book("AAPL", 150.50, 3.0);
    update2.timestamp = update1.timestamp + std::chrono::microseconds(100);
    auto result2 = detector_->process_update(update2);
    
    ASSERT_TRUE(result2.has_value());
    EXPECT_TRUE(result2->rapid_change_detected);
    EXPECT_GT(result2->change_rate, 1000.0); // High rate of change
    
    // Another rapid flip to heavy ask
    auto update3 = create_imbalanced_book("AAPL", 150.50, 0.33);
    update3.timestamp = update2.timestamp + std::chrono::microseconds(100);
    auto result3 = detector_->process_update(update3);
    
    ASSERT_TRUE(result3.has_value());
    EXPECT_TRUE(result3->rapid_change_detected);
    EXPECT_TRUE(result3->is_anomaly);
    EXPECT_EQ(result3->anomaly_type, AnomalyType::RAPID_BOOK_FLIP);
}

// Test multiple symbols
TEST_F(OrderBookImbalanceTest, MultipleSymbols) {
    const std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL"};
    std::map<std::string, double> imbalances;
    
    // Create different imbalances for each symbol
    for (size_t i = 0; i < symbols.size(); ++i) {
        double ratio = 1.0 + i * 0.5; // 1.0, 1.5, 2.0
        auto update = create_imbalanced_book(symbols[i], 100.0 + i * 50, ratio);
        
        auto result = detector_->process_update(update);
        ASSERT_TRUE(result.has_value());
        
        imbalances[symbols[i]] = result->imbalance_ratio;
    }
    
    // Verify each symbol maintains its own state
    for (const auto& symbol : symbols) {
        auto current = detector_->get_current_imbalance(symbol);
        ASSERT_TRUE(current.has_value());
        EXPECT_NEAR(current->imbalance_ratio, imbalances[symbol], 0.01);
    }
}

// Test concurrent updates
TEST_F(OrderBookImbalanceTest, ConcurrentUpdates) {
    const int num_threads = 4;
    const int updates_per_thread = 10000;
    
    std::vector<std::thread> threads;
    std::atomic<int> anomalies_detected{0};
    
    auto worker = [&](int thread_id) {
        utils::set_cpu_affinity(thread_id);
        
        std::string symbol = "SYM" + std::to_string(thread_id);
        
        for (int i = 0; i < updates_per_thread; ++i) {
            // Alternate between balanced and imbalanced
            auto update = (i % 10 < 5) 
                ? create_balanced_book(symbol, 100.0 + i * 0.01)
                : create_imbalanced_book(symbol, 100.0 + i * 0.01, 2.5);
            
            auto result = detector_->process_update(update);
            if (result && result->is_anomaly) {
                anomalies_detected++;
            }
        }
    };
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Should detect roughly half as anomalies
    int expected_anomalies = num_threads * updates_per_thread * 0.5;
    EXPECT_NEAR(anomalies_detected.load(), expected_anomalies, expected_anomalies * 0.1);
}

// Benchmark different book depths
TEST_F(OrderBookImbalanceTest, BenchmarkBookDepths) {
    struct TestCase {
        const char* name;
        size_t depth;
    };
    
    TestCase cases[] = {
        {"1-level", 1},
        {"5-level", 5},
        {"10-level", 10}
    };
    
    for (const auto& tc : cases) {
        // Create book with specified depth
        std::vector<std::pair<double, uint32_t>> bids, asks;
        for (size_t i = 0; i < tc.depth; ++i) {
            bids.push_back({150.0 - 0.01 * i, 1000});
            asks.push_back({150.0 + 0.01 * i, 1000});
        }
        
        auto update = create_book_update("TEST", bids, asks);
        
        // Warm up
        for (int i = 0; i < 10000; ++i) {
            detector_->process_update(update);
        }
        
        // Measure
        const int iterations = 100000;
        uint64_t start = rdtsc();
        for (int i = 0; i < iterations; ++i) {
            update.sequence_number = i;
            detector_->process_update(update);
        }
        uint64_t end = rdtsc();
        
        double avg_cycles = static_cast<double>(end - start) / iterations;
        double avg_ns = avg_cycles / 3.0;
        
        std::cout << "Order Book Imbalance - " << tc.name 
                  << ": " << avg_ns << " ns per update\n";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
